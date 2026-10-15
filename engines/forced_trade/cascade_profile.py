"""
engines/forced_trade/cascade_profile.py

Cycle 66 T3 -- what a labelled cascade actually looks like in the spot tape,
against matched controls. Gated on T1's cluster count.

EVENT SET
---------
BTC/ETH clusters at the primary gap, scheduled segment, uncensored, at or
above the T3 notional floor. The manual 62A cascade is reported separately
("with manual" rows): its price/flow/book profile is legitimate because trades
and order_book_snapshots are continuous and independent of the liquidation
collector, but it never feeds a control.

ORIENTATION
-----------
Everything is signed by the FORCED flow direction d (-1 forced selling, i.e.
longs liquidated; +1 forced buying). A positive oriented price move is a move
WITH the forced flow; a positive REVERSION is a move back against it.

TIMING HONESTY
--------------
A cluster's end is only knowable `gap_sec` after its last liquidation (that is
the silence that closes it), and its size only then. Reversion is therefore
measured from t0 = end + gap_sec (first knowable instant) as the primary
number, and from the raw end as a look-ahead reference only.

GROSS FIRST
-----------
Reversion is reported in gross bps against matched controls (same UTC hour,
same trailing-vol quintile, same tape, observed and cascade-free), then net of
a round-trip cost grid. No strategy, no position sizing, no Sharpe.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from engines.forced_trade.common import ms_to_str

logger = logging.getLogger("forced_trade.cascade_profile")

STEP_MS = 10_000
KEY_OFFSETS_SEC = (-600, -300, -60, 0, 10, 30, 60, 120, 300, 900, 1800, 3600)
HORIZONS_MIN = (1, 5, 15, 30, 60, 240)
COSTS_BPS = (2.0, 9.0, 20.0, 30.0)
COST_LABELS = {2.0: "maker~1bp/side", 9.0: "HL taker (Cycle 57)",
               20.0: "Binance spot taker 10bp/side", 30.0: "Cycle 55 real-money est."}


# ================================================================ series ====

def build_series(buckets: pd.DataFrame, book: pd.DataFrame, book_tol_ms: int = 15_000) -> dict:
    """Price, signed flow and book on one 10s grid.

    Grid point i is the bucket [t_i, t_i + 10s). Price is the bucket's last
    trade (forward-filled through empty buckets); flow is buy minus sell
    aggressor quote in the bucket (0 when empty); book is the last snapshot at
    or before the bucket END, if no older than `book_tol_ms` (else NaN -- the
    book collector has its own outages, which must read as missing, not calm).
    """
    b = buckets.sort_values("bucket_start_ms")
    first = int(b["bucket_start_ms"].iloc[0])
    last = int(b["bucket_start_ms"].iloc[-1])
    n = (last - first) // STEP_MS + 1
    idx = ((b["bucket_start_ms"].values - first) // STEP_MS).astype("int64")
    px = np.full(n, np.nan)
    px[idx] = b["close_price"].values
    px = pd.Series(px).ffill().values
    flow = np.zeros(n)
    flow[idx] = (b["buy_quote"] - b["sell_quote"]).values
    t_end = first + np.arange(n, dtype="int64") * STEP_MS + STEP_MS
    bk = book.sort_values("timestamp")
    bts = bk["timestamp"].values.astype("int64")
    j = np.searchsorted(bts, t_end, side="right") - 1
    ok = (j >= 0) & ((t_end - bts[np.clip(j, 0, len(bts) - 1)]) <= book_tol_ms)
    jj = np.clip(j, 0, len(bts) - 1)

    def take(col):
        v = np.full(n, np.nan)
        v[ok] = bk[col].values.astype(float)[jj[ok]]
        return v

    return {"first": first, "n": n, "px": px, "flow": flow,
            "depth": take("bid_volume_top10") + take("ask_volume_top10"),
            "bid": take("bid_volume_top10"), "ask": take("ask_volume_top10"),
            "spread": take("spread_bps")}


def idx_of(series: dict, t_ms) -> np.ndarray:
    return ((np.asarray(t_ms, dtype="int64") - series["first"]) // STEP_MS).astype("int64")


def take_at(arr: np.ndarray, i: np.ndarray) -> np.ndarray:
    out = np.full(i.shape, np.nan)
    ok = (i >= 0) & (i < len(arr))
    out[ok] = arr[i[ok]]
    return out


# ============================================================== profiles ===

def profiles(series: dict, anchors_ms, directions, offsets_sec=KEY_OFFSETS_SEC,
             baseline_hours: float = 24.0) -> dict:
    """Oriented event-time profiles, one row per anchor.

    price  -- d * log(px[t] / px[anchor - 60s]) in bps
    flow   -- d * (buy - sell aggressor quote) in the 10s bucket, USD
    depth / hit-side depth / spread -- ratio to the anchor's own trailing
              `baseline_hours` median (strictly before the anchor)
    """
    a = np.asarray(anchors_ms, dtype="int64")
    d = np.asarray(directions, dtype=float)
    i0 = idx_of(series, a)
    offs = np.asarray(offsets_sec) * 1000 // STEP_MS
    I = i0[:, None] + offs[None, :]
    ref = take_at(series["px"], i0 - 6)
    out = {"offsets_sec": np.asarray(offsets_sec)}
    with np.errstate(divide="ignore", invalid="ignore"):
        out["price_bps"] = d[:, None] * np.log(take_at(series["px"], I) / ref[:, None]) * 1e4
    out["flow_usd"] = d[:, None] * take_at(series["flow"], I)
    nb = int(baseline_hours * 3_600_000 // STEP_MS)
    base_depth, base_spread, base_hit = [], [], []
    for k, (i, di) in enumerate(zip(i0, d)):
        lo = max(0, i - nb)
        seg = slice(lo, max(lo, i))
        base_depth.append(np.nanmedian(series["depth"][seg]) if i > lo else np.nan)
        base_spread.append(np.nanmedian(series["spread"][seg]) if i > lo else np.nan)
        hit = series["bid"] if di < 0 else series["ask"]
        base_hit.append(np.nanmedian(hit[seg]) if i > lo else np.nan)
    base_depth, base_spread, base_hit = map(np.asarray, (base_depth, base_spread, base_hit))
    with np.errstate(divide="ignore", invalid="ignore"):
        out["depth_ratio"] = take_at(series["depth"], I) / base_depth[:, None]
        out["spread_ratio"] = take_at(series["spread"], I) / base_spread[:, None]
        hit_now = np.where(d[:, None] < 0, take_at(series["bid"], I), take_at(series["ask"], I))
        out["hit_depth_ratio"] = hit_now / base_hit[:, None]
    return out


def profile_table(ev_prof: dict, ct_prof: dict, ct_event_idx: np.ndarray) -> pd.DataFrame:
    """Median event value vs median of per-event control means, per offset."""
    rows = []
    offs = ev_prof["offsets_sec"]
    for ch in ("price_bps", "flow_usd", "depth_ratio", "hit_depth_ratio", "spread_ratio"):
        ev = ev_prof[ch]
        ct = pd.DataFrame(ct_prof[ch]).groupby(ct_event_idx).mean().reindex(range(len(ev))).values
        for j, o in enumerate(offs):
            e, c = ev[:, j], ct[:, j]
            ok = np.isfinite(e) & np.isfinite(c)
            rows.append({"channel": ch, "offset_sec": int(o),
                         "event_median": float(np.nanmedian(e)) if np.isfinite(e).any() else np.nan,
                         "control_median": float(np.nanmedian(c)) if np.isfinite(c).any() else np.nan,
                         "median_excess": float(np.median(e[ok] - c[ok])) if ok.any() else np.nan,
                         "n": int(ok.sum())})
    return pd.DataFrame(rows)


# ============================================================= reversion ===

def moves(series: dict, start_ms, end_ms, directions, gap_sec: float,
          horizons_min=HORIZONS_MIN) -> pd.DataFrame:
    """Cascade move and reversion, oriented, in bps.

    move          -- d * (px[end] / px[start - 60s] - 1)
    rev_<h>m      -- -d * (px[t0 + h] / px[t0] - 1), t0 = end + gap_sec
                     (first instant the cluster is known to be over)
    rev_<h>m_end  -- same from the raw end (look-ahead reference only)
    """
    s = idx_of(series, start_ms)
    e = idx_of(series, end_ms)
    d = np.asarray(directions, dtype=float)
    g = int(round(gap_sec * 1000 / STEP_MS))
    px = series["px"]
    out = {"move_bps": d * (take_at(px, e) / take_at(px, s - 6) - 1) * 1e4}
    for h in horizons_min:
        k = h * 60_000 // STEP_MS
        t0 = e + g
        out[f"rev_{h}m"] = -d * (take_at(px, t0 + k) / take_at(px, t0) - 1) * 1e4
        out[f"rev_{h}m_end"] = -d * (take_at(px, e + k) / take_at(px, e) - 1) * 1e4
    return pd.DataFrame(out)


def day_block_ci(values: np.ndarray, days: np.ndarray, n_boot: int = 2000,
                 seed: int = 66, q=(0.05, 0.95)) -> tuple:
    """90% interval for the mean, resampling whole UTC days (events on the
    same day share a regime and are not independent draws)."""
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    v, dd = v[ok], np.asarray(days)[ok]
    if len(v) < 3:
        return (np.nan, np.nan)
    u, inv = np.unique(dd, return_inverse=True)
    sums = np.bincount(inv, weights=v, minlength=len(u))
    cnts = np.bincount(inv, minlength=len(u)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), size=(n_boot, len(u)))
    means = sums[idx].sum(1) / np.maximum(cnts[idx].sum(1), 1)
    return float(np.quantile(means, q[0])), float(np.quantile(means, q[1]))


def day_cluster_se(values: np.ndarray, days: np.ndarray) -> float:
    """Standard error of the mean with UTC days as clusters (events on one day
    share a regime): sqrt(G/(G-1) * sum_g (sum_{i in g} (v_i - mean))^2) / n."""
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    v, dd = v[ok], np.asarray(days)[ok]
    if len(v) < 3:
        return np.nan
    _, inv = np.unique(dd, return_inverse=True)
    g = int(inv.max()) + 1
    if g < 2:
        return np.nan
    s = np.bincount(inv, weights=v - v.mean(), minlength=g)
    return float(np.sqrt(g / (g - 1) * (s ** 2).sum()) / len(v))


def reversion_table(ev_moves: pd.DataFrame, ct_moves: pd.DataFrame, ct_event_idx,
                    days, costs=COSTS_BPS) -> pd.DataFrame:
    """Gross first: event mean, matched-control mean, excess with a day-block
    CI, hit rate, and the dispersion behind them (SD across events, SE iid and
    day-clustered). Then the same event mean net of each round-trip cost."""
    ctm = ct_moves.groupby(np.asarray(ct_event_idx)).mean().reindex(range(len(ev_moves)))
    rows = []
    for col in ev_moves.columns:
        ev = ev_moves[col].values.astype(float)
        ct = ctm[col].values.astype(float)
        ex = ev - ct
        lo, hi = day_block_ci(ex, days)
        n_ex = int(np.isfinite(ex).sum())
        sd_ex = float(np.nanstd(ex, ddof=1)) if n_ex > 1 else np.nan
        r = {"metric": col, "n": int(np.isfinite(ev).sum()),
             "event_mean_bps": float(np.nanmean(ev)), "event_median_bps": float(np.nanmedian(ev)),
             "control_mean_bps": float(np.nanmean(ct)), "excess_mean_bps": float(np.nanmean(ex)),
             "excess_ci90_lo": lo, "excess_ci90_hi": hi,
             "event_hit_rate": float(np.nanmean(ev > 0)) if np.isfinite(ev).any() else np.nan,
             "event_sd_bps": float(np.nanstd(ev, ddof=1)) if np.isfinite(ev).sum() > 1 else np.nan,
             "excess_sd_bps": sd_ex,
             "excess_se_iid_bps": sd_ex / np.sqrt(n_ex) if n_ex > 1 else np.nan,
             "excess_se_day_bps": day_cluster_se(ex, days),
             "n_days": int(len(np.unique(np.asarray(days)[np.isfinite(ex)])))}
        if col.startswith("rev_") and not col.endswith("_end"):
            for c in costs:
                r[f"net_{c:g}bps_mean"] = float(np.nanmean(ev - c))
                r[f"net_{c:g}bps_hit"] = float(np.nanmean((ev - c) > 0))
        rows.append(r)
    return pd.DataFrame(rows)


def event_frame(cl: pd.DataFrame) -> pd.DataFrame:
    e = cl.reset_index(drop=True).copy()
    e["day"] = (e["start_ms"] // 86_400_000).astype("int64")
    e["start_utc"] = e["start_ms"].map(ms_to_str)
    return e
