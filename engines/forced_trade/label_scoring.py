"""
engines/forced_trade/label_scoring.py

Cycle 66 T2 -- score the Cycle 61 A2 detector, AS BUILT, against Bybit
liquidation labels.

THE DETECTOR IS NOT TOUCHED
---------------------------
Detection goes through cascade.to_windows -> cascade.detect ->
cascade.merge_events with cascade.DEFAULT_SWEEP, exactly the calls
cascade.run_detector makes. The only new input is a bucket cache that covers
the labelled window. `regression_check` proves both halves of "unchanged":
the Cycle 61 cache through today's code must reproduce the Cycle 61 counts
(code unchanged), and today's cache truncated at the Cycle 61 boundary must
reproduce them too (data unchanged).

WHAT IS SCORED, AND AGAINST WHAT
--------------------------------
Labels are gap-based clusters (label_inventory.cluster_labels). "Label
present" means a cluster at or above a notional floor; BTC averages roughly one
liquidation a minute, so "any liquidation nearby" would be true most of the
time and would make precision meaningless. Every precision is printed next to
its chance rate and the lift over it.

  window level   -- the detector's own 60s windows inside observed time: the
                    2x2, precision, recall, FPR, prevalence, lift.
  detector->label-- each detector event: is a qualifying cluster within tau?
  label->detector-- each cluster: STRICT (a detector event STARTS within tau of
                    the cluster onset) and OVERLAP (the detector fired anywhere
                    inside the cluster's span). Bucketed by size, because the
                    venue mismatch makes pooled recall uninterpretable: a $10M+
                    cascade should register on any venue.

THE VENUE CONFOUND, STATED ONCE
-------------------------------
Labels are Bybit USDT-linear PERP liquidations; the detector reads Binance SPOT
trades. No Binance perp trade table exists. A Bybit cascade that does not
transmit to Binance spot is invisible to the detector by construction; recall
in the TOP size bucket is the number that separates detector failure from a
venue artefact.

CONTROLS
--------
Every "at labelled events" statistic has a control drawn ONLY from the
scheduled observed mask (the manual 62A segment never feeds a control or a
chance rate), away from any cluster >= `exclusion_floor_usd`. Two kinds:
uniform (Cycle 61's design, now mask-aware) and matched (same UTC hour, same
trailing-volatility quintile). The matched control is the number to read.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from engines.forced_trade import cascade as C
from engines.forced_trade import corroborate as CO
from engines.forced_trade.common import MS_PER_SEC, ms_to_str
from engines.forced_trade.label_inventory import (
    COUNT_LABELS, SIZE_LABELS, count_bucket, size_bucket,
)
from engines.forced_trade.labels import merge_intervals, points_inside, spans_inside

logger = logging.getLogger("forced_trade.label_scoring")

# Published Cycle 61 counts (RETRO_cycle61_forced_trade_data_audit.md, T1.1).
CYCLE61_COUNTS = {("BTC", "loose"): 2092, ("BTC", "base"): 229, ("BTC", "strict"): 26,
                  ("ETH", "loose"): 1142, ("ETH", "base"): 72, ("ETH", "strict"): 5}


# ============================================================== detection ===

def detect_full(buckets: pd.DataFrame, p: C.CascadeParams, bucket_sec: int = 10) -> tuple:
    """Exactly cascade.run_detector's calls, but keeping the per-window frame
    (its diagnostic columns are needed to ask what the detector SAW)."""
    w = C.to_windows(buckets, bucket_sec, p.window_sec)
    d = C.detect(w, p)
    ev = C.merge_events(d, p, p.window_sec)
    return d, ev


def regression_check(old_buckets: dict, new_buckets: dict, bucket_sec: int = 10) -> pd.DataFrame:
    """A control that can fail: reproduce the published Cycle 61 counts.

    (a) Cycle 61's own cache through today's detector code -> code unchanged.
    (b) Today's cache truncated at Cycle 61's last bucket -> data unchanged.
    Plus a bucket-level diff of (b) against (a), so a count mismatch can be
    traced to the buckets that caused it.
    """
    rows = []
    for asset in sorted(old_buckets):
        old = old_buckets[asset]
        last_old = int(old["bucket_start_ms"].max())
        new = new_buckets[asset]
        trunc = new[new["bucket_start_ms"] <= last_old]
        m = old.merge(trunc, on="bucket_start_ms", how="outer", suffixes=("_old", "_new"),
                      indicator=True)
        both = m[m["_merge"] == "both"]
        mism = np.zeros(len(both), dtype=bool)
        for col in ("n_trades", "buy_quote", "sell_quote", "close_price"):
            mism |= ~np.isclose(both[f"{col}_old"].astype(float).values,
                                both[f"{col}_new"].astype(float).values,
                                rtol=1e-9, atol=1e-9)
        n_mism = int(mism.sum())
        for name, p in C.DEFAULT_SWEEP:
            _, ev_old = detect_full(old, p, bucket_sec)
            _, ev_new = detect_full(trunc, p, bucket_sec)
            exp = CYCLE61_COUNTS.get((asset, name))
            rows.append({
                "asset": asset, "setting": name, "published": exp,
                "c61_cache_now_code": int(len(ev_old)),
                "c66_cache_truncated": int(len(ev_new)),
                "code_unchanged": int(len(ev_old)) == exp,
                "data_reproduces": int(len(ev_new)) == exp,
                "buckets_only_old": int((m["_merge"] == "left_only").sum()),
                "buckets_only_new": int((m["_merge"] == "right_only").sum()),
                "buckets_value_mismatch": n_mism,
                "c61_last_bucket": ms_to_str(last_old)})
    return pd.DataFrame(rows)


# ============================================================ interval ops ==

def overlaps_any(lo, hi, a, b) -> np.ndarray:
    """Vectorised: does [a, b) intersect any merged interval [lo, hi)?"""
    a = np.asarray(a, dtype="int64")
    b = np.asarray(b, dtype="int64")
    if len(lo) == 0:
        return np.zeros(a.shape, dtype=bool)
    j = np.searchsorted(lo, b, side="left") - 1
    ok = j >= 0
    j = np.clip(j, 0, len(lo) - 1)
    return ok & (hi[j] > a)


def cluster_spans(cl: pd.DataFrame, pad_ms: int = 0):
    """Merged [start - pad, end + pad + 1) spans of a cluster frame."""
    if cl is None or not len(cl):
        return np.array([], dtype="int64"), np.array([], dtype="int64")
    return merge_intervals(cl["start_ms"].values - pad_ms, cl["end_ms"].values + pad_ms + 1)


def event_spans(ev: pd.DataFrame, pad_ms: int = 0):
    if ev is None or not len(ev):
        return np.array([], dtype="int64"), np.array([], dtype="int64")
    return merge_intervals(ev["start_ms"].values - pad_ms, ev["end_ms"].values + pad_ms)


# ======================================================== positive classes ==

def split_clusters(cl: pd.DataFrame, floor_usd: float) -> dict:
    """Scored classes for ONE asset at one floor, scheduled segment only.

    positive  -- uncensored and >= floor, or censored but already >= floor
                 (a lower bound that clears the floor is still a positive)
    ambiguous -- censored and below the floor: might have cleared it if fully
                 observed, so its neighbourhood is excluded from the 2x2
    """
    s = cl[cl["segment"] == "scheduled"]
    big = s["size_usd"] >= floor_usd
    return {"positive": s[big], "ambiguous": s[(~big) & s["censored"]],
            "scored_uncensored": s[~s["censored"]]}


# ===================================================== window-level 2x2 =====

def window_confusion(d: pd.DataFrame, window_sec: int, cl: pd.DataFrame, floor_usd: float,
                     tau_sec: float, mask_lo, mask_hi, n_boot: int = 1000,
                     seed: int = 66) -> dict:
    """The 2x2 on the detector's own windows, inside observed time only."""
    w_ms = window_sec * MS_PER_SEC
    ts = d["ts_ms"].values.astype("int64")
    ready = d["med_dom"].notna().values & d["sd_ret"].notna().values
    inmask = spans_inside(mask_lo, mask_hi, ts, ts + w_ms)
    parts = split_clusters(cl, floor_usd)
    tau = int(tau_sec * 1000)
    p_lo, p_hi = cluster_spans(parts["positive"], tau)
    a_lo, a_hi = cluster_spans(parts["ambiguous"], tau)
    lab = overlaps_any(p_lo, p_hi, ts, ts + w_ms)
    amb = overlaps_any(a_lo, a_hi, ts, ts + w_ms) & (~lab)
    scope = ready & inmask & (~amb)
    det = d["cand"].values.astype(bool)
    y, x = lab[scope], det[scope]
    tp = int((x & y).sum())
    fp = int((x & ~y).sum())
    fn = int((~x & y).sum())
    tn = int((~x & ~y).sum())
    n = tp + fp + fn + tn
    out = {"floor_usd": floor_usd, "tau_sec": tau_sec, "windows": n, "TP": tp, "FP": fp,
           "FN": fn, "TN": tn, "excluded_ambiguous": int((ready & inmask & amb).sum()),
           "n_positive_clusters": int(len(parts["positive"]))}
    prev = (tp + fn) / n if n else np.nan
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    out.update(precision=prec, prevalence=prev, recall=tp / (tp + fn) if (tp + fn) else np.nan,
               fpr=fp / (fp + tn) if (fp + tn) else np.nan,
               lift=prec / prev if (prev and not np.isnan(prec)) else np.nan)
    # Day-block bootstrap: windows within a day are not independent.
    if n and n_boot and (tp + fp):
        day = (ts[scope] // 86_400_000)
        days, inv = np.unique(day, return_inverse=True)
        agg = np.zeros((len(days), 4))
        np.add.at(agg, inv, np.stack([(x & y), (x & ~y), (~x & y), (~x & ~y)], 1).astype(float))
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(days), size=(n_boot, len(days)))
        s = agg[idx].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            bp = s[:, 0] / (s[:, 0] + s[:, 1])
            bprev = (s[:, 0] + s[:, 2]) / s.sum(axis=1)
            bl = bp / bprev
        out["precision_ci90"] = [float(np.nanquantile(bp, 0.05)), float(np.nanquantile(bp, 0.95))]
        out["lift_ci90"] = [float(np.nanquantile(bl, 0.05)), float(np.nanquantile(bl, 0.95))]
        out["n_days"] = int(len(days))
    return out


# ================================================ event-level, both ways ====

def random_placements(durations_ms, mask_lo, mask_hi, n: int, rng) -> tuple:
    """Uniform placements of spans (durations resampled) fully inside the mask."""
    w = (mask_hi - mask_lo).astype(float)
    durs = np.asarray(durations_ms, dtype="int64")
    starts, ends = [], []
    tries = 0
    while len(starts) < n and tries < 50:
        tries += 1
        m = (n - len(starts)) * 2
        k = rng.choice(len(w), size=m, p=w / w.sum())
        t = mask_lo[k] + (rng.random(m) * w[k]).astype("int64")
        dd = durs[rng.integers(0, len(durs), size=m)] if len(durs) else np.zeros(m, "int64")
        ok = spans_inside(mask_lo, mask_hi, t, t + dd)
        starts.extend(t[ok].tolist())
        ends.extend((t + dd)[ok].tolist())
    return np.array(starts[:n], dtype="int64"), np.array(ends[:n], dtype="int64")


def detector_to_label(ev: pd.DataFrame, cl: pd.DataFrame, floor_usd: float, tau_sec: float,
                      mask_lo, mask_hi, n_null: int = 20000, seed: int = 66) -> dict:
    """Precision direction: each detector event inside observed time -- is a
    qualifying cluster within tau? Chance from random placements of the same
    event durations inside the same mask."""
    tau = int(tau_sec * 1000)
    inside = spans_inside(mask_lo, mask_hi, ev["start_ms"].values, ev["end_ms"].values) \
        if len(ev) else np.zeros(0, bool)
    e = ev[inside]
    pos = split_clusters(cl, floor_usd)["positive"]
    p_lo, p_hi = cluster_spans(pos, tau)
    hit = overlaps_any(p_lo, p_hi, e["start_ms"].values, e["end_ms"].values)
    rng = np.random.default_rng(seed)
    durs = (e["end_ms"] - e["start_ms"]).values if len(e) else np.array([60_000])
    rs, re_ = random_placements(durs, mask_lo, mask_hi, n_null, rng)
    chance = float(overlaps_any(p_lo, p_hi, rs, re_).mean()) if len(rs) else np.nan
    prec = float(hit.mean()) if len(e) else np.nan
    dir_agree = np.nan
    if len(e) and hit.any():
        # side agreement of matched pairs: detector dom_side vs cluster direction
        agree = []
        cs = pos.sort_values("start_ms")
        for _, r in e[hit].iterrows():
            near = cs[(cs["end_ms"] >= r["start_ms"] - tau) & (cs["start_ms"] <= r["end_ms"] + tau)]
            if len(near):
                big = near.loc[near["size_usd"].idxmax()]
                agree.append(int(np.sign(r["dom_side"]) == np.sign(big["direction"])))
        dir_agree = float(np.mean(agree)) if agree else np.nan
    return {"floor_usd": floor_usd, "tau_sec": tau_sec, "events_in_mask": int(len(e)),
            "events_outside_mask": int((~inside).sum()), "matched": int(hit.sum()),
            "precision": prec, "chance": chance,
            "lift": prec / chance if chance else np.nan, "side_agreement": dir_agree}


def label_to_detector(cl_scored: pd.DataFrame, d: pd.DataFrame, ev: pd.DataFrame,
                      window_sec: int, tau_sec: float) -> pd.DataFrame:
    """Recall direction, per cluster: STRICT and OVERLAP hits."""
    w_ms = window_sec * MS_PER_SEC
    tau = int(tau_sec * 1000)
    cand = d.loc[d["cand"].astype(bool), "ts_ms"].values.astype("int64")
    c_lo, c_hi = merge_intervals(cand, cand + w_ms)
    s = cl_scored["start_ms"].values.astype("int64")
    e = cl_scored["end_ms"].values.astype("int64")
    overlap = overlaps_any(c_lo, c_hi, s, e + 1)
    ev_start = np.sort(ev["start_ms"].values.astype("int64")) if len(ev) else np.array([], "int64")
    j = np.searchsorted(ev_start, s - tau, side="left")
    strict = (j < len(ev_start)) & (ev_start[np.clip(j, 0, max(len(ev_start) - 1, 0))]
                                    <= s + tau) if len(ev_start) else np.zeros(len(s), bool)
    out = cl_scored.copy()
    out["hit_overlap"] = overlap
    out["hit_strict"] = strict
    return out


def null_hit_rates(cl_scored: pd.DataFrame, d: pd.DataFrame, ev: pd.DataFrame,
                   window_sec: int, tau_sec: float, mask_lo, mask_hi, k: int = 20,
                   seed: int = 66) -> pd.DataFrame:
    """Chance recall: each cluster's span re-placed k times uniformly inside
    the mask; the mean hit rate is what a detector firing at random would
    score on clusters of that duration."""
    rng = np.random.default_rng(seed)
    w_ms = window_sec * MS_PER_SEC
    tau = int(tau_sec * 1000)
    cand = d.loc[d["cand"].astype(bool), "ts_ms"].values.astype("int64")
    c_lo, c_hi = merge_intervals(cand, cand + w_ms)
    ev_start = np.sort(ev["start_ms"].values.astype("int64")) if len(ev) else np.array([], "int64")
    durs = np.repeat((cl_scored["end_ms"] - cl_scored["start_ms"]).values.astype("int64"), k)
    w = (mask_hi - mask_lo).astype(float)
    kk = rng.choice(len(w), size=len(durs), p=w / w.sum())
    room = np.maximum(w[kk] - durs, 1.0)
    t = mask_lo[kk] + (rng.random(len(durs)) * room).astype("int64")
    ov = overlaps_any(c_lo, c_hi, t, t + durs + 1)
    if len(ev_start):
        j = np.searchsorted(ev_start, t - tau, side="left")
        st = (j < len(ev_start)) & (ev_start[np.clip(j, 0, len(ev_start) - 1)] <= t + tau)
    else:
        st = np.zeros(len(t), dtype=bool)
    out = cl_scored[["start_ms"]].copy()
    out["null_overlap"] = ov.reshape(-1, k).mean(axis=1) if len(t) else []
    out["null_strict"] = st.reshape(-1, k).mean(axis=1) if len(t) else []
    return out


def recall_table(hits: pd.DataFrame, nulls: pd.DataFrame, by: str) -> pd.DataFrame:
    """Recall per bucket next to its chance rate and lift."""
    h = hits.copy()
    h["null_overlap"] = nulls["null_overlap"].values
    h["null_strict"] = nulls["null_strict"].values
    order = list(SIZE_LABELS) if by == "size_bucket" else list(COUNT_LABELS)
    rows = []
    for b in order + ["ALL"]:
        g = h if b == "ALL" else h[h[by] == b]
        if not len(g):
            rows.append({by: b, "n": 0})
            continue
        r = {by: b, "n": int(len(g)),
             "recall_overlap": float(g["hit_overlap"].mean()),
             "null_overlap": float(g["null_overlap"].mean()),
             "recall_strict": float(g["hit_strict"].mean()),
             "null_strict": float(g["null_strict"].mean()),
             "median_duration_sec": float(g["duration_sec"].median())}
        r["lift_overlap"] = r["recall_overlap"] / r["null_overlap"] if r["null_overlap"] else np.nan
        r["lift_strict"] = r["recall_strict"] / r["null_strict"] if r["null_strict"] else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


# ==================================================== what the detector SAW =

def cluster_diagnostics(cl: pd.DataFrame, d: pd.DataFrame, window_sec: int,
                        settings=C.DEFAULT_SWEEP) -> pd.DataFrame:
    """For each cluster: the detector's own statistics over the windows it
    spans -- how close the as-built detector came, and on which condition it
    failed. The statistics (burst_ratio, imbalance, move_z) depend only on W
    and L, not on K/I/M, so one frame serves all three settings."""
    w_ms = window_sec * MS_PER_SEC
    ts = d["ts_ms"].values.astype("int64")
    br = d["burst_ratio"].values
    im = d["imbalance"].values
    mz = d["move_z"].values
    rows = []
    for _, c in cl.iterrows():
        i0 = int(np.searchsorted(ts, int(c["start_ms"]) - w_ms + 1, side="left"))
        i1 = int(np.searchsorted(ts, int(c["end_ms"]), side="right"))
        if i1 <= i0:
            rows.append({})
            continue
        seg_br, seg_im, seg_mz = br[i0:i1], im[i0:i1], mz[i0:i1]
        r = {"windows": i1 - i0,
             "max_burst_ratio": float(np.nanmax(seg_br)) if np.isfinite(seg_br).any() else np.nan,
             "max_abs_imbalance": float(np.nanmax(np.abs(seg_im))),
             "max_abs_move_z": float(np.nanmax(np.abs(seg_mz))) if np.isfinite(seg_mz).any()
             else np.nan}
        k = int(np.nanargmax(np.where(np.isfinite(seg_br), seg_br, -np.inf)))
        r["imb_sign_at_max_burst_agrees"] = bool(np.sign(seg_im[k]) == np.sign(c["direction"]))
        for name, p in settings:
            okK = seg_br > p.burst_k
            okI = np.abs(seg_im) > p.imbalance_i
            okM = np.abs(seg_mz) > p.move_m
            r[f"{name}_anyK"] = bool(np.nanmax(okK)) if len(okK) else False
            r[f"{name}_anyI"] = bool(np.nanmax(okI)) if len(okI) else False
            r[f"{name}_anyM"] = bool(np.nanmax(okM)) if len(okM) else False
            r[f"{name}_all3"] = bool(np.nanmax(okK & okI & okM)) if len(okK) else False
            # DIAGNOSTIC ONLY: K and M in the SAME window, i.e. what blocked the
            # as-built rule was I alone. Not a detector, never scored.
            r[f"{name}_KM_same"] = bool(np.nanmax(okK & okM)) if len(okK) else False
        rows.append(r)
    return pd.concat([cl.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def null_diagnostics(durations_ms, d: pd.DataFrame, window_sec: int, mask_lo, mask_hi,
                     n: int = 4000, seed: int = 66) -> pd.DataFrame:
    """The same three statistics at random observed spans of the same lengths."""
    rng = np.random.default_rng(seed)
    s, e = random_placements(durations_ms, mask_lo, mask_hi, n, rng)
    fake = pd.DataFrame({"start_ms": s, "end_ms": e, "direction": rng.choice([-1, 1], len(s))})
    return cluster_diagnostics(fake, d, window_sec, settings=())


# ======================================================== controls ==========

def trailing_vol(grid: dict, lookback_min: int = 60) -> np.ndarray:
    """Strictly trailing realised vol of 10s log returns over `lookback_min`."""
    px = pd.Series(grid["px"])
    r = np.log(px).diff()
    n = int(lookback_min * 60_000 // grid["step"])
    return r.rolling(n, min_periods=n // 2).std().shift(1).values


def control_pool(mask_lo, mask_hi, exclusion_lo, exclusion_hi, step_ms: int = 60_000,
                 span_ms: int = 60_000) -> np.ndarray:
    """Candidate control anchors: every `step_ms` instant whose
    [t, t + span_ms] is observed and not within any exclusion interval."""
    out = []
    for a, b in zip(mask_lo, mask_hi):
        first = ((a + step_ms - 1) // step_ms) * step_ms
        t = np.arange(first, b - span_ms, step_ms, dtype="int64")
        if len(t):
            out.append(t)
    t = np.concatenate(out) if out else np.array([], dtype="int64")
    if len(exclusion_lo):
        t = t[~overlaps_any(exclusion_lo, exclusion_hi, t, t + span_ms)]
    return t


def matched_controls(anchors_ms, pool_ms, grid: dict, vol: np.ndarray, k: int = 20,
                     n_vol_buckets: int = 5, seed: int = 66) -> pd.DataFrame:
    """k controls per anchor: same UTC hour-of-day, same trailing-vol quintile.

    Quintile edges come from the POOL (observed, cascade-free time), so an
    event's vol is ranked against ordinary conditions, not against other
    events. Cells too thin to supply k distinct controls are flagged.
    """
    rng = np.random.default_rng(seed)

    def vol_at(t):
        i = (np.asarray(t, dtype="int64") - grid["first"]) // grid["step"]
        i = np.clip(i, 0, grid["n"] - 1)
        return vol[i]

    pv = vol_at(pool_ms)
    good = np.isfinite(pv)
    pool, pv = pool_ms[good], pv[good]
    edges = np.quantile(pv, np.linspace(0, 1, n_vol_buckets + 1)[1:-1])
    p_hour = (pool // 3_600_000) % 24
    p_vb = np.searchsorted(edges, pv, side="right")
    cells = {}
    for h, v, t in zip(p_hour, p_vb, pool):
        cells.setdefault((int(h), int(v)), []).append(int(t))
    cells = {kk: np.array(vv, dtype="int64") for kk, vv in cells.items()}
    rows = []
    av = vol_at(anchors_ms)
    for i, (t, v) in enumerate(zip(np.asarray(anchors_ms, dtype="int64"), av)):
        h = int((t // 3_600_000) % 24)
        vb = int(np.searchsorted(edges, v, side="right")) if np.isfinite(v) else -1
        cand = cells.get((h, vb), np.array([], dtype="int64"))
        thin = len(cand) < k
        pick = rng.choice(cand, size=min(k, len(cand)), replace=False) if len(cand) else []
        for c in pick:
            rows.append({"event_idx": i, "anchor_ms": int(c), "hour": h, "vol_bucket": vb,
                         "cell_size": int(len(cand)), "thin_cell": thin})
    return pd.DataFrame(rows)


def uniform_controls(n_events: int, pool_ms, k: int = 20, seed: int = 66) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = max(500, k * n_events)
    pick = rng.choice(pool_ms, size=min(n, len(pool_ms)), replace=False)
    return pd.DataFrame({"anchor_ms": np.sort(pick)})


# ================================================== book stress =============

def depth_side_ratio(spans: pd.DataFrame, book: pd.DataFrame, baseline_hours: float = 24.0,
                     min_baseline_snaps: int = 30) -> pd.DataFrame:
    """Hit-side depth vs its own trailing baseline: BID depth under forced
    selling, ASK depth under forced buying. Same baseline rule as
    corroborate.depth_profile (median over the preceding window)."""
    ts = book["timestamp"].values.astype("int64")
    bid = book["bid_volume_top10"].values.astype(float)
    ask = book["ask_volume_top10"].values.astype(float)
    base_ms = int(baseline_hours * 3_600_000)
    out = []
    for _, e in spans.iterrows():
        lo, hi = int(e["start_ms"]), int(e["end_ms"])
        series = bid if e["direction"] < 0 else ask
        i0, i1 = np.searchsorted(ts, lo), np.searchsorted(ts, hi)
        b0, b1 = np.searchsorted(ts, lo - base_ms), i0
        if i1 > i0 and (b1 - b0) >= min_baseline_snaps:
            base = float(np.nanmedian(series[b0:b1]))
            out.append(float(np.nanmean(series[i0:i1])) / base if base else np.nan)
        else:
            out.append(np.nan)
    return pd.DataFrame({"hit_side_depth_ratio": out})


def stress_spans(cl: pd.DataFrame, min_dur_sec: float = 60.0) -> pd.DataFrame:
    """Cluster spans padded to at least one detector window, so a single-event
    cluster is measured over the same length as a Cycle 61 detector event."""
    s = cl["start_ms"].values.astype("int64")
    e = np.maximum(cl["end_ms"].values.astype("int64"), s + int(min_dur_sec * 1000))
    return pd.DataFrame({"start_ms": s, "end_ms": e, "duration_sec": (e - s) / 1000.0,
                         "direction": cl["direction"].values})


def book_stress(spans: pd.DataFrame, controls: pd.DataFrame, book: pd.DataFrame,
                baseline_hours: float = 24.0) -> dict:
    """Depth / spread / hit-side depth at the spans and at their controls.

    Control spans reuse the matched event's duration and direction (for the
    hit side). The depth and spread ratios come from Cycle 61's own
    corroborate.depth_profile so the numbers are directly comparable with its
    depth 1.02 / spread 1.00.
    """
    prof = CO.depth_profile(spans, book, baseline_hours=baseline_hours)
    side = depth_side_ratio(spans, book, baseline_hours)
    cprof = CO.depth_profile(controls, book, baseline_hours=baseline_hours)
    cside = depth_side_ratio(controls, book, baseline_hours)
    out = {}
    for metric, ev_s, ct_s in (
            ("depth", prof.get("depth_ratio"), cprof.get("depth_ratio")),
            ("spread_bps", prof.get("spread_ratio"), cprof.get("spread_ratio")),
            ("hit_side_depth", side["hit_side_depth_ratio"], cside["hit_side_depth_ratio"])):
        out[metric] = CO.summarise_ratio(metric, ev_s if ev_s is not None else pd.Series(dtype=float),
                                         ct_s if ct_s is not None else pd.Series(dtype=float))
    return out


def control_spans_for(events: pd.DataFrame, ctrl: pd.DataFrame) -> pd.DataFrame:
    """Give each control the duration and direction of the event it matches."""
    if "event_idx" in ctrl.columns:
        dur = (events["end_ms"] - events["start_ms"]).values[ctrl["event_idx"].values]
        dirn = events["direction"].values[ctrl["event_idx"].values]
    else:
        rng = np.random.default_rng(66)
        j = rng.integers(0, len(events), size=len(ctrl))
        dur = (events["end_ms"] - events["start_ms"]).values[j]
        dirn = events["direction"].values[j]
    s = ctrl["anchor_ms"].values.astype("int64")
    return pd.DataFrame({"start_ms": s, "end_ms": s + dur, "duration_sec": dur / 1000.0,
                         "direction": dirn})


def add_buckets(cl: pd.DataFrame) -> pd.DataFrame:
    out = cl.copy()
    out["size_bucket"] = size_bucket(out["size_usd"].values)
    out["count_bucket"] = count_bucket(out["n_events"].values)
    return out


def points_in_mask(t, lo, hi) -> np.ndarray:
    return points_inside(lo, hi, t)
