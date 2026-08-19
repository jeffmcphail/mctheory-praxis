"""
engines/forced_trade/corroborate.py

T1 (second half) -- corroboration and the HONEST false-positive assessment for
the liquidation-cascade detector.

WHY THIS MODULE IS SEPARATE FROM cascade.py
-------------------------------------------
cascade.py finds bursts of one-sided aggressive flow. That is all it does, and a
burst of one-sided aggressive flow is not a liquidation cascade -- it is a burst
of one-sided aggressive flow. Everything that argues those detections are
actually cascades lives here, clearly labelled as circumstantial, because there
is no liquidation feed to label against and therefore NO TRUE FALSE-POSITIVE
RATE IS COMPUTABLE. Saying so is the deliverable; a fabricated precision number
would not be.

WHAT IS ACTUALLY MEASURABLE
---------------------------
1. DEPTH COLLAPSE (order_book_snapshots). A forced-liquidation cascade eats
   resting depth. If detections show no depth withdrawal relative to a matched
   random control, the detector is finding volume, not stress.
2. CROSS-ASSET CONCORDANCE. BTC and ETH detectors run on completely separate
   order flow. A market-wide deleveraging hits both within minutes; an
   idiosyncratic block trade hits one. Measured against the rate expected by
   chance, this gives a LIFT that no single-asset statistic can.
3. TEMPORAL CONCENTRATION. Real cascades cluster into a handful of stress days.
   A detector firing uniformly across the calendar is a volume-burst detector.
   Quantified as a dispersion index against a Poisson null (index 1.0).
4. EXTREME-DAY OVERLAP. Do detections land on the largest absolute daily-return
   days? Reported WITH its circularity flagged: the detector already conditions
   on a price move, so agreement here is partly mechanical.

THE RANDOM CONTROL IS THE POINT
-------------------------------
Every "at events" statistic below is reported next to the same statistic
computed at random non-event timestamps drawn from the same span. Without the
control, "median depth ratio 0.93 during events" is unreadable -- it could be
the unconditional value of the ratio.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from engines.forced_trade.common import (
    DEFAULT_DB, MS_PER_DAY, MS_PER_HOUR, MS_PER_SEC, ms_to_str, read_only_db,
)

logger = logging.getLogger("forced_trade.corroborate")

_BOOK_COLS = ("timestamp", "mid_price", "spread_bps", "bid_volume_top10",
              "ask_volume_top10", "order_imbalance_top10")


# ============================================================ book loading ==

def load_book(asset, db_path=DEFAULT_DB, first_ms=None, last_ms=None) -> pd.DataFrame:
    """Load order-book snapshots for one asset (about 1M rows -- fits in memory)."""
    sql = f"SELECT {', '.join(_BOOK_COLS)} FROM order_book_snapshots WHERE asset=?"
    params = [asset]
    if first_ms is not None:
        sql += " AND timestamp >= ?"
        params.append(int(first_ms))
    if last_ms is not None:
        sql += " AND timestamp <= ?"
        params.append(int(last_ms))
    sql += " ORDER BY timestamp"
    with read_only_db(db_path) as conn:
        df = pd.read_sql_query(sql, conn, params=params)
    df["depth"] = df["bid_volume_top10"] + df["ask_volume_top10"]
    logger.info("[%s] loaded %d book snapshots %s -> %s", asset, len(df),
                ms_to_str(df["timestamp"].min()) if len(df) else "-",
                ms_to_str(df["timestamp"].max()) if len(df) else "-")
    return df


def book_cadence(book: pd.DataFrame) -> dict:
    """Sampling cadence of the snapshot series -- sets the resolution floor.

    A detector window shorter than the median snapshot gap cannot be
    corroborated by the book at all, and that has to be stated rather than
    silently produce thin per-event means.
    """
    if len(book) < 2:
        return {"n": len(book)}
    gaps = np.diff(book["timestamp"].values) / MS_PER_SEC
    return {
        "n": int(len(book)),
        "median_gap_sec": float(np.median(gaps)),
        "p90_gap_sec": float(np.percentile(gaps, 90)),
        "p99_gap_sec": float(np.percentile(gaps, 99)),
        "max_gap_sec": float(np.max(gaps)),
        "frac_gaps_over_60s": float(np.mean(gaps > 60)),
    }


# ======================================================== depth around ev ==

def _window_stats(book_ts, book_vals, lo_ms, hi_ms):
    """Mean of `book_vals` for snapshots in [lo_ms, hi_ms). NaN if none."""
    i0 = np.searchsorted(book_ts, lo_ms, side="left")
    i1 = np.searchsorted(book_ts, hi_ms, side="left")
    if i1 <= i0:
        return np.nan, 0
    return float(np.nanmean(book_vals[i0:i1])), int(i1 - i0)


def depth_profile(events: pd.DataFrame, book: pd.DataFrame,
                  baseline_hours: float = 24.0,
                  min_baseline_snaps: int = 30) -> pd.DataFrame:
    """Per-event book depth and spread vs their own trailing baseline.

    Baseline is the MEDIAN over the `baseline_hours` immediately BEFORE the
    event -- strictly trailing, so the event cannot contaminate its own
    reference.
    """
    cols = ["depth_at_event", "depth_baseline", "depth_ratio",
            "spread_at_event", "spread_baseline", "spread_ratio",
            "n_snaps_event", "n_snaps_baseline"]
    if events.empty or book.empty:
        return pd.DataFrame(columns=cols)

    ts = book["timestamp"].values
    depth = book["depth"].values
    spread = book["spread_bps"].values
    base_ms = int(baseline_hours * MS_PER_HOUR)

    rows = []
    for _, e in events.iterrows():
        lo, hi = int(e["start_ms"]), int(e["end_ms"])
        d_ev, n_ev = _window_stats(ts, depth, lo, hi)
        s_ev, _ = _window_stats(ts, spread, lo, hi)

        b0 = np.searchsorted(ts, lo - base_ms, side="left")
        b1 = np.searchsorted(ts, lo, side="left")
        n_base = int(b1 - b0)
        if n_base >= min_baseline_snaps:
            d_base = float(np.nanmedian(depth[b0:b1]))
            s_base = float(np.nanmedian(spread[b0:b1]))
        else:
            d_base = s_base = np.nan

        rows.append({
            "depth_at_event": d_ev, "depth_baseline": d_base,
            "depth_ratio": d_ev / d_base if d_base else np.nan,
            "spread_at_event": s_ev, "spread_baseline": s_base,
            "spread_ratio": s_ev / s_base if s_base else np.nan,
            "n_snaps_event": n_ev, "n_snaps_baseline": n_base,
        })
    return pd.DataFrame(rows, columns=cols)


def random_control(events: pd.DataFrame, book: pd.DataFrame,
                   first_ms: int, last_ms: int, n_draws: int = 500,
                   baseline_hours: float = 24.0, seed: int = 61,
                   exclusion_sec: int = 900) -> pd.DataFrame:
    """The same depth/spread statistic at random NON-event timestamps.

    Durations are resampled from the observed event durations so the control
    windows are the same length as the real ones -- a shorter window has fewer
    snapshots and a noisier mean, which would otherwise bias the comparison.
    Draws within `exclusion_sec` of any detected event are rejected.
    """
    if book.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    durations = (events["duration_sec"].values if len(events)
                 else np.array([60]))
    ex = int(exclusion_sec * MS_PER_SEC)
    ev_lo = events["start_ms"].values.astype("int64") - ex if len(events) else np.array([])
    ev_hi = events["end_ms"].values.astype("int64") + ex if len(events) else np.array([])

    lo_bound = int(first_ms + baseline_hours * MS_PER_HOUR)
    picks = []
    attempts = 0
    while len(picks) < n_draws and attempts < n_draws * 50:
        attempts += 1
        t = int(rng.integers(lo_bound, max(lo_bound + 1, last_ms)))
        if len(ev_lo) and np.any((t >= ev_lo) & (t <= ev_hi)):
            continue
        dur = int(rng.choice(durations))
        picks.append({"start_ms": t, "end_ms": t + dur * MS_PER_SEC,
                      "duration_sec": dur})
    ctrl = pd.DataFrame(picks)
    logger.debug("random_control: %d draws accepted in %d attempts", len(ctrl), attempts)
    return depth_profile(ctrl, book, baseline_hours=baseline_hours)


def summarise_ratio(name: str, at_events: pd.Series, at_control: pd.Series) -> dict:
    """Median / IQR of a ratio at events vs at the matched random control."""
    ev = pd.to_numeric(at_events, errors="coerce").dropna()
    ct = pd.to_numeric(at_control, errors="coerce").dropna()
    out = {"metric": name, "n_events": int(len(ev)), "n_control": int(len(ct))}
    if len(ev):
        out.update(ev_median=float(ev.median()),
                   ev_p25=float(ev.quantile(0.25)),
                   ev_p75=float(ev.quantile(0.75)),
                   ev_frac_below_1=float((ev < 1).mean()))
    if len(ct):
        out.update(ctrl_median=float(ct.median()),
                   ctrl_frac_below_1=float((ct < 1).mean()))
    if len(ev) and len(ct) and ct.median():
        out["ev_over_ctrl"] = float(ev.median() / ct.median())
    return out


# ================================================= false-positive evidence ==

def daily_counts(events: pd.DataFrame, first_ms: int, last_ms: int) -> pd.Series:
    """Events per UTC day over the FULL span (zero-filled)."""
    days = pd.date_range(pd.Timestamp(first_ms, unit="ms", tz="UTC").normalize(),
                         pd.Timestamp(last_ms, unit="ms", tz="UTC").normalize(),
                         freq="D")
    if events.empty:
        return pd.Series(0, index=days, dtype=int)
    d = pd.to_datetime(events["start_ms"], unit="ms", utc=True).dt.normalize()
    return d.value_counts().reindex(days, fill_value=0).sort_index()


def concentration(counts: pd.Series) -> dict:
    """Is the detector firing in clusters, or uniformly across the calendar?

    Dispersion index = var/mean. A Poisson process gives 1.0. Real cascades are
    clustered, so >> 1 is the signature we want; ~1 says the detector is finding
    something that happens at a constant background rate.
    """
    n = int(counts.sum())
    out = {"n_events": n, "n_days": int(len(counts)),
           "days_with_event": int((counts > 0).sum())}
    if n == 0 or counts.mean() == 0:
        return out
    out["mean_per_day"] = float(counts.mean())
    out["dispersion_index"] = float(counts.var(ddof=1) / counts.mean())
    order = counts.sort_values(ascending=False)
    for k in (1, 3, 5, 10):
        if len(order) >= k:
            out[f"share_top{k}_days"] = float(order.iloc[:k].sum() / n)
    out["max_day_count"] = int(counts.max())
    out["max_day"] = str(counts.idxmax().date())
    return out


def cross_asset_concordance(ev_a: pd.DataFrame, ev_b: pd.DataFrame,
                            first_ms: int, last_ms: int,
                            tol_sec: int = 300) -> dict:
    """Fraction of A-events with a B-event within tol, and the LIFT over chance.

    The chance rate is the probability that a randomly placed +/-tol window
    catches at least one B event, approximated as the fraction of the span
    covered by B-event neighbourhoods. Lift = observed / chance. Lift near 1
    means the two detectors are independent, i.e. they are finding
    asset-idiosyncratic bursts, not market-wide deleveraging.
    """
    out = {"n_a": int(len(ev_a)), "n_b": int(len(ev_b)), "tol_sec": tol_sec}
    if ev_a.empty or ev_b.empty:
        out["matched_frac"] = float("nan")
        return out
    tol = tol_sec * MS_PER_SEC
    a = np.sort(ev_a["start_ms"].values.astype("int64"))
    b_lo = ev_b["start_ms"].values.astype("int64") - tol
    b_hi = ev_b["end_ms"].values.astype("int64") + tol

    order = np.argsort(b_lo)
    b_lo, b_hi = b_lo[order], b_hi[order]
    matched = 0
    for t in a:
        j = np.searchsorted(b_lo, t, side="right") - 1
        if j >= 0 and t <= b_hi[:j + 1].max():
            matched += 1
    out["matched"] = int(matched)
    out["matched_frac"] = float(matched / len(a))

    # union coverage of B neighbourhoods over the span -> chance rate
    span = max(last_ms - first_ms, 1)
    merged, cur_lo, cur_hi = 0, None, None
    for lo, hi in zip(b_lo, b_hi):
        if cur_lo is None:
            cur_lo, cur_hi = lo, hi
        elif lo <= cur_hi:
            cur_hi = max(cur_hi, hi)
        else:
            merged += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
    if cur_lo is not None:
        merged += cur_hi - cur_lo
    chance = merged / span
    out["chance_frac"] = float(chance)
    out["lift"] = float(out["matched_frac"] / chance) if chance > 0 else float("nan")
    return out


def extreme_day_overlap(events: pd.DataFrame, daily: pd.DataFrame,
                        top_n: int = 10, first_ms=None, last_ms=None) -> dict:
    """Do detections land on the largest absolute daily-return days?

    THE WINDOW MUST MATCH THE EVENTS. `ohlcv_daily` runs from 2023-11 while the
    tick data starts 2026-04-29, so ranking over the whole daily history picks
    the ten biggest movers of a 2.7-year sample -- almost none of which fall in
    the 111-day window where events can exist. That comparison returns zero
    overlap BY CONSTRUCTION and would read as "the detector avoids stress days",
    which is the opposite of what it shows. The daily frame is therefore
    restricted to the event span before ranking.

    PARTIALLY CIRCULAR even when correct, and reported as such: the detector
    already conditions on a price move (condition M), so some agreement is
    mechanical. It is still informative in one direction -- if detections did
    NOT concentrate on the big days, the detector would be finding noise.
    """
    if events.empty or daily.empty:
        return {"top_n": top_n}
    d = daily.copy()
    d["abs_ret"] = (d["close"] / d["close"].shift(1) - 1).abs()
    if first_ms is not None:
        d = d[d["timestamp"] >= int(first_ms)]
    if last_ms is not None:
        d = d[d["timestamp"] <= int(last_ms)]
    d = d.dropna(subset=["abs_ret"])
    n_days = len(d)
    if n_days == 0:
        return {"top_n": top_n, "error": "no daily bars inside the event window"}
    top_n = min(top_n, n_days)
    big = set(d.nlargest(top_n, "abs_ret")["date"].astype(str))
    ev_days = pd.to_datetime(events["start_ms"], unit="ms", utc=True).dt.date.astype(str)
    hit = int(ev_days.isin(big).sum())
    chance = top_n / n_days
    return {
        "top_n": top_n,
        "n_days_in_window": n_days,
        "n_events": int(len(events)),
        "events_on_top_days": hit,
        "frac_on_top_days": float(hit / len(events)),
        "chance_frac": float(chance),
        "lift": float((hit / len(events)) / chance) if chance > 0 else float("nan"),
        "top_days": sorted(big),
        "circularity_note": ("detector condition M already requires an abnormal "
                             "price move, so agreement here is partly mechanical"),
    }


def load_daily(asset, db_path=DEFAULT_DB) -> pd.DataFrame:
    with read_only_db(db_path) as conn:
        return pd.read_sql_query(
            "SELECT date, timestamp, open, high, low, close, volume FROM ohlcv_daily "
            "WHERE asset=? ORDER BY timestamp", conn, params=(asset,))
