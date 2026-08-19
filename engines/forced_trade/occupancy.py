"""
engines/forced_trade/occupancy.py

T5 -- scenario x regime grid feasibility.

WHAT "12 REGIME CLASSES" MEANS HERE
-----------------------------------
`docs/REGIME_MATRIX.md` defines twelve regime CLASSES A-L, each with its own
3-5 discrete states. Those are twelve AXES, not twelve cells: the full joint
product is
    5 * 4 * 3 * 5 * 3 * 5 * 4 * 3 * 4 * 3 * 3 * 3  =  about 10.5 million cells,
which no event count in crypto can populate. So occupancy is reported at the
only two granularities that can be interpreted:

    MARGINAL   scenario x (class, state) -- 45 cells per scenario, one table per
               axis. This is what "events per regime class" can mean when the
               event count is in the hundreds.
    JOINT-2    scenario x (vol level B) x (liquidity G) -- the coarse collapse
               the brief asks for, 4 x 4 = 16 cells, reported alongside a 3 x 3
               tercile version.

Reporting a "12-class grid" as if it were 12 cells would be the laundering the
taxonomy warns about, so the arithmetic is stated instead.

CAUSALITY
---------
Regime is evaluated on a fixed TRAILING window ending at each evaluation
timestamp, and an event is assigned the most recent evaluation STRICTLY AT OR
BEFORE its start. `RegimeEngine.compute_time_series` instead feeds an
ever-growing prefix (`ohlcv_hourly.iloc[:end_idx]`), which makes the early
evaluations use less history than the late ones and slowly changes what a state
means over the sample. For an occupancy census that inconsistency would smear
the counts, so this module runs its own fixed-window loop over the same
`RegimeEngine.compute` entry point.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from engines.forced_trade.common import (
    DEFAULT_DB, MS_PER_DAY, MS_PER_HOUR, ms_to_str, read_only_db,
)

logger = logging.getLogger("forced_trade.occupancy")

OCCUPANCY_BINS = ((0, 0, "0"), (1, 2, "1-2"), (3, 9, "3-9"), (10, 10**9, "10+"))


@dataclass(frozen=True)
class OccupancyParams:
    trailing_days: int = 90       # fixed regime lookback (bounded, consistent)
    step_hours: int = 24          # regime re-evaluation cadence
    bars_per_day: int = 24        # crypto
    min_events_estimable: int = 10  # what counts as an estimable cell


# ============================================================ data loading ==

def load_hourly(asset, db_path=DEFAULT_DB) -> pd.DataFrame:
    """Hourly OHLCV built from `ohlcv_1m` (the only intraday source we hold)."""
    with read_only_db(db_path) as conn:
        m = pd.read_sql_query(
            "SELECT timestamp, open, high, low, close, volume FROM ohlcv_1m "
            "WHERE asset=? ORDER BY timestamp", conn, params=(asset,))
    if m.empty:
        return m
    m["dt"] = pd.to_datetime(m["timestamp"], unit="ms", utc=True)
    h = m.set_index("dt").resample("1h").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"), volume=("volume", "sum"))
    h = h.dropna(subset=["close"])
    logger.info("[%s] hourly bars: %d  %s -> %s", asset, len(h),
                h.index[0], h.index[-1])
    return h


def load_funding(asset, venue="binance", db_path=DEFAULT_DB) -> pd.DataFrame:
    with read_only_db(db_path) as conn:
        f = pd.read_sql_query(
            "SELECT timestamp, funding_rate FROM funding_rates "
            "WHERE asset=? AND venue=? ORDER BY timestamp",
            conn, params=(asset, venue))
    if not f.empty:
        f["dt"] = pd.to_datetime(f["timestamp"], unit="ms", utc=True)
    return f


# ========================================================== regime series ==

def regime_series(asset, hourly, funding=None, universe=None,
                  p: OccupancyParams = OccupancyParams(),
                  first_ms=None, last_ms=None) -> pd.DataFrame:
    """Regime state at each evaluation point, on a FIXED trailing window.

    Returns one row per evaluation: `ts_ms` plus one integer column per class.
    `n_missing` records how many of the twelve classes the engine could not
    compute -- a class that silently defaults to state 0 would otherwise inflate
    the 0-state bucket and make the grid look more occupied than it is.
    """
    from engines.regime_engine import RegimeEngine, REGIME_CLASSES

    eng = RegimeEngine(bars_per_day=p.bars_per_day)
    win = p.trailing_days * 24
    if len(hourly) <= win:
        raise ValueError(f"[{asset}] {len(hourly)} hourly bars <= trailing window "
                         f"{win}; cannot evaluate regime")

    idx = hourly.index
    start_i = win
    if first_ms is not None:
        want = pd.Timestamp(first_ms, unit="ms", tz="UTC")
        start_i = max(win, int(idx.searchsorted(want)))
    end_i = len(hourly)
    if last_ms is not None:
        want = pd.Timestamp(last_ms, unit="ms", tz="UTC")
        end_i = min(end_i, int(idx.searchsorted(want)) + 1)

    rows = []
    for i in range(start_i, end_i, p.step_hours):
        chunk = hourly.iloc[i - win:i]
        fr = None
        if funding is not None and not funding.empty:
            fr = funding.loc[funding["dt"] < idx[i - 1], "funding_rate"].values
            if fr.size < 3:
                fr = None
        uni = None
        if universe:
            uni = {a: df.iloc[max(0, i - win):i] for a, df in universe.items()
                   if len(df) >= i}
        try:
            st = eng.compute(ohlcv_hourly=chunk, funding_rates=fr,
                             universe_ohlcv=uni)
        except Exception as e:  # noqa: BLE001
            logger.warning("[%s] regime compute failed at %s: %s", asset, idx[i - 1], e)
            continue
        row = {"ts_ms": int(idx[i - 1].value // 10**6),
               "n_missing": len(st.missing),
               "missing": ",".join(st.missing)}
        for c in REGIME_CLASSES:
            row[f"class_{c}"] = int(st.states.get(c, 0))
        rows.append(row)

    df = pd.DataFrame(rows)
    logger.info("[%s] regime series: %d evaluations %s -> %s "
                "(mean classes missing %.2f)", asset, len(df),
                ms_to_str(df["ts_ms"].iloc[0]) if len(df) else "-",
                ms_to_str(df["ts_ms"].iloc[-1]) if len(df) else "-",
                df["n_missing"].mean() if len(df) else float("nan"))
    return df


def axis_degeneracy(regimes: pd.DataFrame) -> pd.DataFrame:
    """How many states does each class ACTUALLY take, given the data we hold?

    This is the number a grid designer needs and the one the matrix does not
    supply. `docs/REGIME_MATRIX.md` declares 3-5 states per class, but a class
    whose inputs are missing does not announce itself -- it returns a constant
    and is not listed in `RegimeState.missing`. An axis stuck on one value adds
    no partitions, so counting it toward "12 regime classes" overstates the
    grid by exactly the factor of its declared states.
    """
    from engines.regime_engine import (REGIME_CLASSES, REGIME_CLASS_NAMES,
                                       REGIME_STATE_RANGES)
    miss_counts = {}
    if "missing" in regimes:
        for s in regimes["missing"].fillna(""):
            for c in [x for x in str(s).split(",") if x]:
                miss_counts[c] = miss_counts.get(c, 0) + 1

    rows = []
    n = len(regimes)
    for c in REGIME_CLASSES:
        col = f"class_{c}"
        vals = regimes[col].dropna().astype(int) if col in regimes else pd.Series(dtype=int)
        obs = sorted(vals.unique().tolist())
        declared = list(REGIME_STATE_RANGES[c])
        mode_share = float(vals.value_counts(normalize=True).iloc[0]) if len(vals) else float("nan")
        rows.append({
            "class": c,
            "class_name": REGIME_CLASS_NAMES[c],
            "declared_states": len(declared),
            "observed_states": len(obs),
            "observed_values": obs,
            "modal_share": round(mode_share, 3),
            "n_evals_missing": miss_counts.get(c, 0),
            "pct_evals_missing": round(100.0 * miss_counts.get(c, 0) / max(n, 1), 1),
            "degenerate": len(obs) <= 1,
        })
    return pd.DataFrame(rows)


def effective_grid_size(deg: pd.DataFrame) -> dict:
    """Nominal vs effective grid size once degenerate axes are discounted."""
    nominal_joint = int(np.prod(deg["declared_states"].values))
    obs = deg["observed_states"].clip(lower=1).values
    effective_joint = int(np.prod(obs))
    return {
        "nominal_joint_cells": nominal_joint,
        "effective_joint_cells": effective_joint,
        "nominal_marginal_cells": int(deg["declared_states"].sum()),
        "effective_marginal_cells": int(deg["observed_states"].sum()),
        "degenerate_axes": deg.loc[deg["degenerate"], "class"].tolist(),
        "informative_axes": int((~deg["degenerate"]).sum()),
    }


def assign_regime(events, regimes) -> pd.DataFrame:
    """Attach the most recent regime evaluation AT OR BEFORE each event start."""
    from engines.regime_engine import REGIME_CLASSES
    cls_cols = [f"class_{c}" for c in REGIME_CLASSES]
    if events.empty or regimes.empty:
        return events.assign(**{c: pd.Series(dtype="float") for c in cls_cols})
    r = regimes.sort_values("ts_ms").reset_index(drop=True)
    idx = np.searchsorted(r["ts_ms"].values, events["start_ms"].values, side="right") - 1
    out = events.copy().reset_index(drop=True)
    ok = idx >= 0
    for c in cls_cols + ["n_missing"]:
        vals = np.full(len(out), np.nan)
        vals[ok] = r[c].values[idx[ok]]
        out[c] = vals
    out["regime_ts_ms"] = np.where(ok, r["ts_ms"].values[np.clip(idx, 0, None)], np.nan)
    n_unassigned = int((~ok).sum())
    if n_unassigned:
        logger.warning("assign_regime: %d/%d events precede the first regime "
                       "evaluation and are UNASSIGNED", n_unassigned, len(out))
    return out


# =============================================================== occupancy ==

def marginal_occupancy(assigned, scenario="A2_cascades") -> pd.DataFrame:
    """One row per (scenario, class, state) with its event count.

    Every DECLARED state appears, including states with zero events -- an empty
    cell is the finding, and dropping it would hide exactly the sparsity the
    brief is asking about.
    """
    from engines.regime_engine import REGIME_CLASSES, REGIME_CLASS_NAMES, REGIME_STATE_RANGES
    rows = []
    for c in REGIME_CLASSES:
        col = f"class_{c}"
        counts = (assigned[col].dropna().astype(int).value_counts()
                  if col in assigned and len(assigned) else pd.Series(dtype=int))
        for s in REGIME_STATE_RANGES[c]:
            rows.append({
                "scenario": scenario,
                "class": c,
                "class_name": REGIME_CLASS_NAMES[c],
                "state": s,
                "n_events": int(counts.get(s, 0)),
            })
    return pd.DataFrame(rows)


def joint_occupancy(assigned, class_x="B", class_y="G",
                    scenario="A2_cascades") -> pd.DataFrame:
    """Two-axis occupancy -- the coarse collapse the brief asks for."""
    from engines.regime_engine import REGIME_STATE_RANGES
    cx, cy = f"class_{class_x}", f"class_{class_y}"
    rows = []
    for sx in REGIME_STATE_RANGES[class_x]:
        for sy in REGIME_STATE_RANGES[class_y]:
            n = 0
            if len(assigned) and cx in assigned and cy in assigned:
                n = int(((assigned[cx] == sx) & (assigned[cy] == sy)).sum())
            rows.append({"scenario": scenario, "x_class": class_x, "x_state": sx,
                         "y_class": class_y, "y_state": sy, "n_events": n})
    return pd.DataFrame(rows)


# Explicit state collapses to a 3-level axis. Written out rather than computed
# so the grouping is auditable: the merge decides the answer, and an implicit
# rule would hide which states were pooled with which.
DEFAULT_COLLAPSE_3 = {
    "B": {0: "low", 1: "mid", 2: "high", 3: "high"},        # vol level
    "G": {0: "tight", 1: "mid", 2: "wide", 3: "wide"},      # liquidity
    "A": {-2: "down", -1: "down", 0: "flat", 1: "up", 2: "up"},
    "I": {0: "low", 1: "mid", 2: "high", 3: "high"},
}


def collapsed_joint_occupancy(assigned, class_x="B", class_y="G",
                              collapse=None, scenario="A2_cascades") -> pd.DataFrame:
    """3 x 3 occupancy -- the coarser collapse the brief asks for.

    The declared-state joint (4 x 4 for B x G) leaves most cells empty simply
    because the top vol/liquidity states are rare. Merging the extreme states
    into one level is the honest way to ask whether ANY two-axis conditioning is
    supportable, as opposed to concluding "not buildable" from a granularity
    nobody would have chosen.
    """
    collapse = collapse or DEFAULT_COLLAPSE_3
    mx, my = collapse[class_x], collapse[class_y]
    lx = list(dict.fromkeys(mx.values()))
    ly = list(dict.fromkeys(my.values()))
    cx, cy = f"class_{class_x}", f"class_{class_y}"
    rows = []
    for sx in lx:
        for sy in ly:
            n = 0
            if len(assigned) and cx in assigned and cy in assigned:
                gx = assigned[cx].map(lambda v: mx.get(int(v)) if pd.notna(v) else None)
                gy = assigned[cy].map(lambda v: my.get(int(v)) if pd.notna(v) else None)
                n = int(((gx == sx) & (gy == sy)).sum())
            rows.append({"scenario": scenario, "x_class": class_x, "x_level": sx,
                         "y_class": class_y, "y_level": sy, "n_events": n})
    return pd.DataFrame(rows)


def bucket_occupancy(occ: pd.DataFrame) -> pd.DataFrame:
    """Distribution of cell counts into the 0 / 1-2 / 3-9 / 10+ buckets."""
    rows = []
    for lo, hi, label in OCCUPANCY_BINS:
        n = int(((occ["n_events"] >= lo) & (occ["n_events"] <= hi)).sum())
        rows.append({"bucket": label, "n_cells": n,
                     "pct_cells": round(100.0 * n / max(len(occ), 1), 1)})
    return pd.DataFrame(rows)


def granularity_verdict(occ: pd.DataFrame, p: OccupancyParams = OccupancyParams()) -> dict:
    """Plain buildable / not-buildable read at the granularity `occ` describes."""
    total = len(occ)
    estimable = int((occ["n_events"] >= p.min_events_estimable).sum())
    empty = int((occ["n_events"] == 0).sum())
    return {
        "cells": total,
        "cells_empty": empty,
        "cells_estimable": estimable,
        "pct_estimable": round(100.0 * estimable / max(total, 1), 1),
        "min_events_estimable": p.min_events_estimable,
        "buildable": estimable >= 0.5 * total,
    }
