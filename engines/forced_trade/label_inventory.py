"""
engines/forced_trade/label_inventory.py

Cycle 66 T1 analyses on top of labels.py: run accounting, mask validation,
clusters, and the bankruptcy-price bias.

Every function here is a pure computation on the input snapshot -- no database
access -- so the whole inventory can be recomputed at any hour, including the
03:00 mirror / 03:30 backup windows.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from engines.forced_trade.common import MS_PER_SEC, ms_to_str
from engines.forced_trade.labels import (
    SYMBOL_ASSET, Coverage, merge_intervals, points_inside, spans_inside,
)

logger = logging.getLogger("forced_trade.label_inventory")

H_MS = 3_600_000


# ========================================================== run accounting ==

def run_table(cov: Coverage) -> pd.DataFrame:
    rows = []
    for r in cov.runs:
        rows.append({
            "run": r.idx, "log_line": r.line_no, "start_utc": ms_to_str(r.start_ms),
            "start_ms": r.start_ms, "first_connect_ms": r.first_connect_ms,
            "window_end_ms": r.window_end_ms, "last_flush_ms": r.last_flush_ms,
            "exit_code": r.exit_code, "status": r.status,
            "control_frames": r.control_frames,
            "connects": sum(1 for k, _, _ in r.events if k == "connect"),
            "disconnects": sum(1 for k, _, _ in r.events if k == "gap_open"),
            "observed_sec": round(sum(b - a for a, b in r.segments) / 1000.0, 1),
        })
    return pd.DataFrame(rows)


def missing_slots(cov: Coverage, early_sec: int = 120) -> tuple:
    """Hourly slots with no run start, merged into outages.

    A run fills slot H if it started in [H - early_sec, H + 3600 - early_sec),
    so a late StartWhenAvailable start (e.g. 14:42) still fills its hour.
    Each outage records the status of the run before it: a run that hangs past
    the hour makes Task Scheduler (MultipleInstances IgnoreNew) skip the next
    trigger silently, which would pair a killed run with a missing slot.
    """
    first = (cov.first_ms // H_MS) * H_MS
    last = (cov.last_ms // H_MS) * H_MS
    slots = np.arange(first, last, H_MS, dtype="int64")
    filled = {((r.start_ms + early_sec * 1000) // H_MS) * H_MS for r in cov.runs}
    miss = [int(s) for s in slots if int(s) not in filled]
    by_start = sorted(cov.runs, key=lambda r: r.start_ms)

    outages, cur = [], None
    for s in miss:
        if cur and s == cur["last_slot_ms"] + H_MS:
            cur["last_slot_ms"] = s
            cur["hours"] += 1
        else:
            cur = {"first_slot_ms": s, "last_slot_ms": s, "hours": 1}
            outages.append(cur)
    for o in outages:
        prev = [r for r in by_start if r.start_ms < o["first_slot_ms"]]
        nxt = [r for r in by_start
               if r.start_ms >= o["last_slot_ms"] + H_MS - early_sec * 1000]
        o["first_slot_utc"] = ms_to_str(o["first_slot_ms"])
        o["last_slot_utc"] = ms_to_str(o["last_slot_ms"])
        o["prev_run_status"] = prev[-1].status if prev else None
        o["prev_run_log_line"] = prev[-1].line_no if prev else None
        o["prev_run_last_flush_utc"] = (ms_to_str(prev[-1].last_flush_ms)
                                        if prev and prev[-1].last_flush_ms else None)
        o["next_run_start_utc"] = ms_to_str(nxt[0].start_ms) if nxt else None
    return pd.DataFrame({"slot_ms": miss}), pd.DataFrame(outages), int(len(slots))


def handoff_holes(cov: Coverage, max_hole_sec: float = 300.0) -> dict:
    """Unobserved time between consecutive runs (last segment end -> next
    segment start), restricted to holes under `max_hole_sec` so missing runs
    are not counted twice."""
    rs = sorted([r for r in cov.runs if r.segments], key=lambda r: r.start_ms)
    holes = []
    for a, b in zip(rs, rs[1:]):
        h = (b.segments[0][0] - a.segments[-1][1]) / 1000.0
        if 0 <= h <= max_hole_sec:
            holes.append(h)
    h = np.array(holes)
    if not len(h):
        return {"n": 0}
    return {"n": int(len(h)), "total_hours": float(h.sum() / 3600.0),
            "p50_sec": float(np.median(h)), "p95_sec": float(np.quantile(h, 0.95)),
            "max_sec": float(h.max())}


# ===================================================== gap cross-checks =====

def merge_gap_chains(gaps: pd.DataFrame, chain_max_sec: float = 600.0) -> pd.DataFrame:
    """collector_gaps rows -> outages, with retry chains merged.

    The collector overwrites its in-memory open-gap pointer on every failed
    reconnect (engines/bybit_liquidation_collector.py:410-424), so a retry
    chain leaves its earlier rows OPEN and only the last one closed. Read
    literally an open row means "unobserved until now"; it actually ends
    where the chain's closing row ends.
    """
    g = gaps.sort_values("timestamp").reset_index(drop=True)
    out, chain = [], []
    for row in g.itertuples(index=False):
        if chain and (row.timestamp - chain[0].timestamp) / 1000.0 > chain_max_sec:
            for c in chain:
                out.append({"start_ms": int(c.timestamp), "end_ms": None, "rows": 1,
                            "open_rows": 1, "reason": "UNRESOLVED " + str(c.reason)[:60]})
            chain = []
        chain.append(row)
        if pd.notna(row.gap_end):
            out.append({"start_ms": int(chain[0].timestamp), "end_ms": int(row.gap_end),
                        "rows": len(chain), "open_rows": len(chain) - 1,
                        "reason": str(chain[0].reason)[:80]})
            chain = []
    for c in chain:
        out.append({"start_ms": int(c.timestamp), "end_ms": None, "rows": 1,
                    "open_rows": 1, "reason": "UNRESOLVED " + str(c.reason)[:60]})
    df = pd.DataFrame(out)
    if len(df):
        df["seconds"] = (pd.to_numeric(df["end_ms"]) - df["start_ms"]) / 1000.0
    return df


def log_outages(cov: Coverage) -> pd.DataFrame:
    """In-run outages straight from the log: first GAP OPENED -> GAP CLOSED."""
    rows = []
    for r in cov.runs:
        opened = None
        for k, t, x in r.events:
            if k == "gap_open" and opened is None:
                opened = t
            elif k == "gap_close" and opened is not None:
                rows.append({"run": r.idx, "start_ms": opened, "end_ms": x})
                opened = None
        if opened is not None:
            rows.append({"run": r.idx, "start_ms": opened, "end_ms": None})
    return pd.DataFrame(rows)


def gap_crosscheck(chains: pd.DataFrame, log_out: pd.DataFrame, tol_ms: int = 2000) -> dict:
    """Do the merged collector_gaps chains and the log's outages agree?"""
    if chains.empty or log_out.empty:
        return {"chains": int(len(chains)), "log_outages": int(len(log_out))}
    a = chains.sort_values("start_ms").reset_index(drop=True)
    b = log_out.sort_values("start_ms").reset_index(drop=True)
    bs = b["start_ms"].values.astype("int64")
    matched, d_start, d_end = 0, [], []
    for row in a.itertuples(index=False):
        j = int(np.searchsorted(bs, row.start_ms))
        best = None
        for k in (j - 1, j):
            if 0 <= k < len(bs) and abs(int(bs[k]) - int(row.start_ms)) <= tol_ms:
                best = k
        if best is None:
            continue
        matched += 1
        d_start.append(abs(int(bs[best]) - int(row.start_ms)))
        e = b["end_ms"].iloc[best]
        if pd.notna(e) and pd.notna(row.end_ms):
            d_end.append(abs(int(e) - int(row.end_ms)))
    return {"chains": int(len(a)), "log_outages": int(len(b)), "matched": matched,
            "max_start_diff_ms": int(max(d_start)) if d_start else None,
            "max_end_diff_ms": int(max(d_end)) if d_end else None,
            "unresolved_chains": int(pd.to_numeric(a["end_ms"]).isna().sum()),
            "open_rows_absorbed": int(a["open_rows"].sum())}


# ======================================================= mask validation ====

def outside_mask_rows(liq_sched: pd.DataFrame, cov: Coverage, disconnect_pad_ms: int,
                      tol_ms: int = 2000) -> dict:
    """THE MASK VALIDATION: scheduled rows the log mask calls unobserved.

    A row is proof the feed was up at that instant. Rows inside the deliberate
    pre-disconnect pad are expected (the pad is conservative by design). Rows
    within `tol_ms` of a mask edge are second-rounding of the log's local
    timestamps. ANY other row outside the mask means the mask is WRONG.
    """
    t = liq_sched["timestamp"].values.astype("int64")
    inside = points_inside(cov.log_lo, cov.log_hi, t)
    pads = [(g - disconnect_pad_ms, g) for r in cov.runs
            for k, g, _ in r.events if k == "gap_open"]
    p_lo, p_hi = merge_intervals([a for a, _ in pads], [b for _, b in pads])
    in_pad = points_inside(p_lo, p_hi, t)
    n_lo, n_hi = merge_intervals(cov.log_lo - tol_ms, cov.log_hi + tol_ms)
    near = points_inside(n_lo, n_hi, t)
    unexplained = (~inside) & (~in_pad) & (~near)
    return {"scheduled_rows": int(len(t)), "outside_log_mask": int((~inside).sum()),
            "outside_in_disconnect_pad": int(((~inside) & in_pad).sum()),
            "outside_within_tol_of_edge": int(((~inside) & (~in_pad) & near).sum()),
            "outside_UNEXPLAINED": int(unexplained.sum()),
            "unexplained_examples": liq_sched.loc[unexplained, "timestamp"]
            .head(10).map(ms_to_str).tolist(),
            "tol_ms": tol_ms, "disconnect_pad_ms": disconnect_pad_ms}


def disconnect_latency(liq_sched: pd.DataFrame, cov: Coverage, seed: int = 66,
                       n_random: int = 4000) -> dict:
    """How long before each logged disconnect did rows stop arriving?

    Set against the same statistic at random observed instants, because
    liquidations are bursty and "time since the last row" is long anyway. If
    the pre-disconnect lag is much longer than the random lag, the socket was
    dead before the disconnect was logged -- which is what the pad assumes.
    """
    t = np.sort(liq_sched["timestamp"].values.astype("int64"))
    g = np.array([x for r in cov.runs for k, x, _ in r.events if k == "gap_open"],
                 dtype="int64")
    if not len(g) or not len(t):
        return {}
    i = np.searchsorted(t, g, side="left") - 1
    lag = (g[i >= 0] - t[i[i >= 0]]) / 1000.0
    rng = np.random.default_rng(seed)
    w = (cov.log_hi - cov.log_lo).astype(float)
    k = rng.choice(len(w), size=n_random, p=w / w.sum())
    rt = cov.log_lo[k] + (rng.random(n_random) * w[k]).astype("int64")
    j = np.searchsorted(t, rt, side="left") - 1
    rl = (rt[j >= 0] - t[j[j >= 0]]) / 1000.0
    q = (0.25, 0.5, 0.75)
    return {"n_disconnects": int(len(lag)),
            "pre_disconnect_lag_sec_p25_p50_p75": [float(np.quantile(lag, x)) for x in q],
            "random_instant_lag_sec_p25_p50_p75": [float(np.quantile(rl, x)) for x in q]}


def handoff_validation(liq_sched: pd.DataFrame, cov: Coverage, bin_sec: int = 10) -> tuple:
    """Rows per second-of-hour bin against the time the LOG mask calls observed.

    If the mask is right, rows per observed hour is flat across the hour while
    the raw row count collapses exactly where the band sits. Exposure comes
    from the log mask BEFORE the band is applied, so this tests the log
    reconstruction itself, not the band.
    """
    first, last = cov.first_ms, cov.last_ms
    n_sec = int((last - first) // 1000) + 1
    obs = np.zeros(n_sec, dtype=bool)
    for a, b in zip(cov.log_lo, cov.log_hi):
        i0 = max(0, int((a - first + 999) // 1000))
        i1 = min(n_sec, int((b - first) // 1000))
        if i1 > i0:
            obs[i0:i1] = True
    secs = (first // 1000) + np.arange(n_sec, dtype="int64")
    nb = 3600 // bin_sec
    soh_bin = (secs % 3600) // bin_sec
    exposure = np.bincount(soh_bin[obs], minlength=nb) / 3600.0
    all_hours = np.bincount(soh_bin, minlength=nb) / 3600.0
    t = liq_sched["timestamp"].values.astype("int64")
    rows = np.bincount(((t // 1000) % 3600) // bin_sec, minlength=nb)
    df = pd.DataFrame({"soh_start_sec": np.arange(nb) * bin_sec, "rows": rows,
                       "exposure_hours": exposure, "calendar_hours": all_hours})
    df["rows_per_exposed_hour"] = df["rows"] / df["exposure_hours"].replace(0, np.nan)
    df["exposure_frac"] = df["exposure_hours"] / df["calendar_hours"].replace(0, np.nan)
    band = cov.band
    df["in_band"] = ((df["soh_start_sec"] + bin_sec > band["start_soh"])
                     | (df["soh_start_sec"] < band["end_soh"]))
    out = df[~df["in_band"]]
    rate = float(out["rows"].sum() / out["exposure_hours"].sum())
    inb = df[df["in_band"]]
    summ = {
        "bin_sec": bin_sec,
        "band": [band["start_soh"], band["end_soh"]],
        "rate_rows_per_exposed_hour_outside_band": rate,
        "rows_in_band": int(inb["rows"].sum()),
        "rows_expected_in_band_if_collector_had_run": float(rate * inb["calendar_hours"].sum()),
        "rows_expected_in_band_given_log_exposure": float(rate * inb["exposure_hours"].sum()),
        "band_exposure_frac": float(inb["exposure_hours"].sum() / inb["calendar_hours"].sum()),
        "outside_band_exposure_frac":
            float(out["exposure_hours"].sum() / out["calendar_hours"].sum()),
        "outside_band_rate_cv": float(out["rows_per_exposed_hour"].std()
                                      / out["rows_per_exposed_hour"].mean()),
    }
    summ["observed_deficit_frac"] = 1.0 - summ["rows_in_band"] / max(
        summ["rows_expected_in_band_if_collector_had_run"], 1e-9)
    summ["mask_predicted_deficit_frac"] = 1.0 - summ["band_exposure_frac"]
    return df, summ


# ============================================ handoff residual: two tests ===
#
# handoff_validation finds a band deficit larger than the log mask predicts
# (0.81 vs 0.67). Two explanations make different predictions:
#   leading edge -- the new run is dead for longer than `connected to` + pad
#                   says, so the band credits observed time that was not;
#   baseline     -- the out-of-band hourly mean is the wrong yardstick for
#                   the minutes around the top of the hour.
# leading_edge_latency tests the first, handoff_local_baseline the second.

def _handoff_runs(cov: Coverage, max_start_soh: float = 120.0, max_hole_sec: float = 300.0):
    """(run, previous run) pairs that are a true on-schedule handoff: the run
    starts in minute :00/:01 and its predecessor completed in minute :58/:59
    less than `max_hole_sec` earlier. Runs with no usable data are dropped."""
    rs = sorted(cov.runs, key=lambda r: r.start_ms)
    out = []
    for prev, r in zip(rs, rs[1:]):
        if r.status in ("subscribe_rejected", "dead_subscription") or r.first_connect_ms is None:
            continue
        if (r.start_ms % H_MS) / 1000.0 >= max_start_soh or prev.window_end_ms is None:
            continue
        if not 3480 <= (prev.window_end_ms % H_MS) / 1000.0 < 3600:
            continue
        if not 0 <= (r.start_ms - prev.window_end_ms) / 1000.0 <= max_hole_sec:
            continue
        out.append((r, prev))
    return out


def leading_edge_latency(liq_sched: pd.DataFrame, cov: Coverage,
                         max_start_soh: float = 120.0, floor_q: float = 0.05,
                         ramp_edges=(-10, -5, 0, 2, 4, 6, 8, 10, 15, 20, 30, 60, 120),
                         ref_sec=(120, 600)) -> tuple:
    """TEST 1: how late does data really start after an hourly handoff?

    Per on-schedule run: the log's start line (cmd %time%, centiseconds), the
    `connected to` line (whole seconds) and the first liquidation row at or
    after the start. first_row - start bounds the dead time from ABOVE (the
    first event may be genuinely late); its left tail across runs is the
    floor. If the floor sits materially past the band's leading edge, the band
    under-credits the outage.

    The ramp is the same question asked of every row rather than the first:
    rows per run-second in bins after the connect line, against the same
    runs' rate `ref_sec` later. Runs with a logged disconnect inside the
    reference window are dropped so an outage cannot read as a ramp.
    """
    t = np.sort(liq_sched["timestamp"].values.astype("int64"))
    pairs = _handoff_runs(cov, max_start_soh)
    rows = []
    for r, prev in pairs:
        end = r.window_end_ms or r.last_flush_ms or (r.start_ms + H_MS)
        i = int(np.searchsorted(t, r.start_ms, side="left"))
        first = int(t[i]) if i < len(t) and t[i] < end else None
        c = r.first_connect_ms
        rows.append({
            "run": r.idx, "start_utc": ms_to_str(r.start_ms), "status": r.status,
            "start_soh": (r.start_ms % H_MS) / 1000.0, "connect_soh": (c % H_MS) / 1000.0,
            "connect_minus_start_sec": (c - r.start_ms) / 1000.0,
            "prev_end_soh": (prev.window_end_ms % H_MS) / 1000.0,
            "first_row_soh": ((first - r.start_ms) + (r.start_ms % H_MS)) / 1000.0
            if first is not None else np.nan,
            "first_row_minus_start_sec": (first - r.start_ms) / 1000.0
            if first is not None else np.nan,
            "first_row_minus_connect_sec": (first - c) / 1000.0 if first is not None else np.nan,
            "gap_in_ref": any(k == "gap_open" and c <= x < c + ref_sec[1] * 1000
                              for k, x, _ in r.events),
        })
    if not rows:
        return pd.DataFrame(), pd.DataFrame(), {"n_handoff_runs": 0}
    df = pd.DataFrame(rows)
    band_end = cov.band.get("end_soh")
    qs = (0.01, floor_q, 0.10, 0.25, 0.50, 0.75)
    f = df["first_row_soh"].dropna()
    summ = {"n_handoff_runs": int(len(df)), "n_with_first_row": int(len(f)),
            "band_end_soh": band_end,
            "connect_soh_q": {f"p{int(q * 100):02d}": float(df["connect_soh"].quantile(q))
                              for q in (0.01, 0.5, 0.99)},
            "connect_minus_start_sec_q": {f"p{int(q * 100):02d}":
                                          float(df["connect_minus_start_sec"].quantile(q))
                                          for q in (0.01, 0.5, 0.99)},
            "first_row_soh_q": {f"p{int(q * 100):02d}": float(f.quantile(q)) for q in qs},
            "first_row_minus_start_sec_q": {
                f"p{int(q * 100):02d}": float(df["first_row_minus_start_sec"].quantile(q))
                for q in qs},
            "first_row_minus_connect_sec_q": {
                f"p{int(q * 100):02d}": float(df["first_row_minus_connect_sec"].quantile(q))
                for q in qs},
            "floor_q": floor_q,
            "floor_first_row_soh": float(f.quantile(floor_q)) if len(f) else None,
            "runs_first_row_before_band_end": int((f < band_end).sum()) if band_end else None}
    summ["floor_past_band_end_sec"] = (summ["floor_first_row_soh"] - band_end
                                       if band_end is not None and len(f) else None)

    # ramp: every row, by seconds after the connect line's floor-second
    clean = [p for p, row in zip(pairs, rows) if not row["gap_in_ref"]]
    edges = np.asarray(ramp_edges, dtype=float)
    counts = np.zeros(len(edges) - 1)
    ref_rows = 0
    for r, _ in clean:
        c = r.first_connect_ms
        rel = (t[(t >= c + edges[0] * 1000) & (t < c + ref_sec[1] * 1000)] - c) / 1000.0
        counts += np.histogram(rel, bins=edges)[0]
        ref_rows += int(((rel >= ref_sec[0]) & (rel < ref_sec[1])).sum())
    n = max(len(clean), 1)
    ref_rate = ref_rows / (n * (ref_sec[1] - ref_sec[0]))
    ramp = pd.DataFrame({"from_sec": edges[:-1], "to_sec": edges[1:], "rows": counts.astype(int),
                         "rows_per_run_sec": counts / (n * np.diff(edges))})
    ramp["rel_to_ref"] = ramp["rows_per_run_sec"] / ref_rate if ref_rate > 0 else np.nan
    summ["ramp_runs"] = len(clean)
    summ["ramp_ref_sec"] = list(ref_sec)
    summ["ramp_ref_rows_per_run_sec"] = ref_rate
    return df, ramp, summ


def handoff_local_baseline(liq_sched: pd.DataFrame, cov: Coverage, adj_secs=(60, 120, 300),
                           n_boot: int = 1000, seed: int = 66, q=(0.05, 0.95)) -> tuple:
    """TEST 2: the band deficit against a LOCAL baseline instead of the hourly mean.

    Expected band rows come from the `w` seconds immediately before and after
    the band on the same hours (same days, same regimes), at 1-second
    resolution against the log mask. The pre-band window is the yardstick for
    the outgoing run's tail, the post-band window for the incoming run's head.
    `ratio` = observed / expected-given-log-exposure: 1.0 means the mask
    accounts for the whole deficit. Units are rows and active seconds (a
    second with >= 1 row): rows are dominated by a few large clusters, active
    seconds much less so. CIs resample whole UTC days; each hour's band and
    its windows stay inside one resampling unit.
    """
    band = cov.band
    s0, s1 = int(np.floor(band["start_soh"])), int(np.ceil(band["end_soh"]))
    first, last = cov.first_ms, cov.last_ms
    n_sec = int((last - first) // 1000) + 1
    obs = np.zeros(n_sec, dtype=bool)
    for a, b in zip(cov.log_lo, cov.log_hi):
        i0 = max(0, int((a - first + 999) // 1000))
        i1 = min(n_sec, int((b - first) // 1000))
        if i1 > i0:
            obs[i0:i1] = True
    secs = (first // 1000) + np.arange(n_sec, dtype="int64")
    soh = secs % 3600
    ti = liq_sched["timestamp"].values.astype("int64") // 1000 - first // 1000
    ti = ti[(ti >= 0) & (ti < n_sec)]
    rows = np.bincount(ti, minlength=n_sec).astype(float)
    units = {"rows": rows, "active_sec": (rows > 0).astype(float)}
    day = (secs + 1800) // 86400
    _, dinv = np.unique(day, return_inverse=True)
    nd = int(dinv.max()) + 1

    tail = soh >= s0
    head = soh < s1
    inb = tail | head
    regions = {"band": inb, "tail": tail, "head": head, "outside": ~inb}
    for w in adj_secs:
        regions[f"pre{w}"] = (soh >= s0 - w) & (soh < s0)
        regions[f"post{w}"] = (soh >= s1) & (soh < s1 + w)

    def per_day(mask, weights):
        return np.bincount(dinv[mask], weights=weights[mask], minlength=nd)

    cal = {k: per_day(m, np.ones(n_sec)) for k, m in regions.items()}
    expo = {k: per_day(m & obs, np.ones(n_sec)) for k, m in regions.items()}
    cnt = {u: {k: per_day(m, v) for k, m in regions.items()} for u, v in units.items()}

    def stats(sel, unit, yard):
        """yard = 'global' or a window width; sel = day multiplicities."""
        S = {k: float((v * sel).sum()) for k, v in expo.items()}
        C = {k: float((v * sel).sum()) for k, v in cnt[unit].items()}
        cal_band = float((cal["band"] * sel).sum())
        if yard == "global":
            r_tail = r_head = r_all = C["outside"] / max(S["outside"], 1e-9)
        else:
            r_tail = C[f"pre{yard}"] / max(S[f"pre{yard}"], 1e-9)
            r_head = C[f"post{yard}"] / max(S[f"post{yard}"], 1e-9)
            r_all = (C[f"pre{yard}"] + C[f"post{yard}"]) / max(S[f"pre{yard}"]
                                                                + S[f"post{yard}"], 1e-9)
        exp_band = r_tail * S["tail"] + r_head * S["head"]
        return {"observed": C["band"], "expected_given_exposure": exp_band,
                "ratio": C["band"] / max(exp_band, 1e-9),
                "ratio_tail": C["tail"] / max(r_tail * S["tail"], 1e-9),
                "ratio_head": C["head"] / max(r_head * S["head"], 1e-9),
                "observed_deficit": 1.0 - C["band"] / max(r_all * cal_band, 1e-9),
                "mask_predicted_deficit": 1.0 - S["band"] / max(cal_band, 1e-9),
                "rate_tail_yard": r_tail, "rate_head_yard": r_head}

    rng = np.random.default_rng(seed)
    boot = [np.bincount(rng.integers(0, nd, nd), minlength=nd).astype(float)
            for _ in range(n_boot)]
    one = np.ones(nd)
    out = []
    for unit in units:
        for yard in ("global",) + tuple(adj_secs):
            st = stats(one, unit, yard)
            bs = [stats(b, unit, yard) for b in boot]
            row = {"unit": unit, "baseline": "global hourly mean" if yard == "global"
                   else f"local +-{yard}s", **st}
            for k in ("ratio", "ratio_tail", "ratio_head", "observed_deficit"):
                v = np.array([x[k] for x in bs])
                row[f"{k}_ci90_lo"], row[f"{k}_ci90_hi"] = (float(np.quantile(v, q[0])),
                                                            float(np.quantile(v, q[1])))
            out.append(row)
    df = pd.DataFrame(out)
    summ = {"band_seconds": [s0, s1], "days": nd, "n_boot": n_boot,
            "band_calendar_hours": float(cal["band"].sum() / 3600.0),
            "band_log_exposed_hours": float(expo["band"].sum() / 3600.0),
            "tail_log_exposed_hours": float(expo["tail"].sum() / 3600.0),
            "head_log_exposed_hours": float(expo["head"].sum() / 3600.0)}
    return df, summ


# ============================================================ spot prices ===

def spot_grid(buckets: pd.DataFrame, bucket_sec: int = 10) -> dict:
    """Full 10s grid of last-trade price (forward-filled) for O(1) lookup."""
    b = buckets.sort_values("bucket_start_ms")
    step = bucket_sec * MS_PER_SEC
    first = int(b["bucket_start_ms"].iloc[0])
    last = int(b["bucket_start_ms"].iloc[-1])
    n = (last - first) // step + 1
    px = np.full(n, np.nan)
    px[((b["bucket_start_ms"].values - first) // step).astype("int64")] = \
        b["close_price"].values
    return {"first": first, "step": step, "px": pd.Series(px).ffill().values, "n": n}


def spot_at(grid: dict, t) -> np.ndarray:
    t = np.asarray(t, dtype="int64")
    i = (t - grid["first"]) // grid["step"]
    ok = (i >= 0) & (i < grid["n"])
    out = np.full(t.shape, np.nan)
    out[ok] = grid["px"][i[ok]]
    return out


# =============================================================== clusters ===

def cluster_labels(liq: pd.DataFrame, gap_sec: float, cov: Coverage,
                   grids: dict) -> pd.DataFrame:
    """Gap-based clusters per symbol: a new cluster starts when the time since
    the previous liquidation ON THE SAME SYMBOL exceeds `gap_sec`.

    Size is reported three ways and they are not interchangeable:
      n_events      -- row count (Bybit has no per-second throttle)
      quote_raw     -- sum of Bybit quote_qty = BANKRUPTCY price x qty (biased)
      notional_spot -- sum of qty x Binance spot last-trade price (BTC/ETH only)
    `censored` marks clusters whose [start - gap, end + gap] is not inside one
    observed interval: the cluster may extend into unobserved time, so its size
    is a lower bound and it is excluded from scored event sets. Manual-segment
    clusters are censored by construction (no pre-event observation).
    """
    gap_ms = int(gap_sec * 1000)
    out = []
    for sym, d in liq.groupby("symbol", sort=True):
        d = d.sort_values("timestamp")
        ts = d["timestamp"].values.astype("int64")
        cid = np.cumsum(np.r_[True, np.diff(ts) > gap_ms]) - 1
        asset = SYMBOL_ASSET.get(sym)
        dirn = d["direction"].values.astype(float)
        qty = d["quantity"].values.astype(float)
        notional = qty * spot_at(grids[asset], ts) if asset in grids else \
            np.full(len(ts), np.nan)
        f = pd.DataFrame({"cid": cid, "ts": ts, "qty": qty,
                          "quote_raw": d["quote_qty"].values.astype(float),
                          "notional_spot": notional, "sell": (dirn < 0).astype(int),
                          "signed_raw": dirn * d["quote_qty"].values.astype(float),
                          "signed_spot": dirn * notional})
        g = f.groupby("cid")
        c = pd.DataFrame({
            "start_ms": g["ts"].min(), "end_ms": g["ts"].max(),
            "n_events": g["ts"].size(), "qty": g["qty"].sum(),
            "quote_raw": g["quote_raw"].sum(),
            "notional_spot": g["notional_spot"].sum(min_count=1),
            "signed_raw": g["signed_raw"].sum(),
            "signed_spot": g["signed_spot"].sum(min_count=1),
            "n_forced_sell": g["sell"].sum(),
        }).reset_index(drop=True)
        c.insert(0, "symbol", sym)
        c.insert(1, "asset", asset)
        c["gap_sec"] = gap_sec
        c["duration_sec"] = (c["end_ms"] - c["start_ms"]) / 1000.0
        # Direction of the FORCED flow: -1 net forced selling (longs
        # liquidated), +1 net forced buying (shorts liquidated).
        c["direction"] = np.where(c["signed_raw"] < 0, -1, 1)
        c["purity"] = (c["signed_raw"].abs()
                       / c["quote_raw"].replace(0, np.nan)).fillna(0.0)
        c["segment"] = np.where(c["start_ms"] < cov.first_ms, "manual", "scheduled")
        sched = (c["segment"] == "scheduled").values
        s, e = c["start_ms"].values, c["end_ms"].values
        c["censored_left"] = np.where(sched, ~spans_inside(cov.lo, cov.hi, s - gap_ms, s), True)
        c["censored_right"] = np.where(sched, ~spans_inside(cov.lo, cov.hi, e, e + gap_ms), True)
        c["censored"] = np.where(sched, ~spans_inside(cov.lo, cov.hi, s - gap_ms, e + gap_ms),
                                 True)
        out.append(c)
    allc = pd.concat(out, ignore_index=True)
    allc["start_utc"] = allc["start_ms"].map(ms_to_str)
    allc["size_usd"] = allc["notional_spot"].where(allc["notional_spot"].notna(),
                                                   allc["quote_raw"])
    return allc


SIZE_EDGES = (1e5, 1e6, 1e7)
SIZE_LABELS = ("<100k", "100k-1M", "1M-10M", ">=10M")
COUNT_EDGES = (5, 20, 100)
COUNT_LABELS = ("1-4", "5-19", "20-99", ">=100")


def size_bucket(usd) -> np.ndarray:
    """Fixed decade buckets on cluster notional (USD), set before any scoring."""
    x = np.asarray(usd, dtype=float)
    idx = np.searchsorted(np.asarray(SIZE_EDGES), np.nan_to_num(x, nan=-1.0), side="right")
    out = np.array([SIZE_LABELS[i] for i in idx], dtype=object)
    out[np.isnan(x)] = "n/a"
    return out


def count_bucket(n) -> np.ndarray:
    idx = np.searchsorted(np.asarray(COUNT_EDGES), np.asarray(n), side="right")
    return np.array([COUNT_LABELS[i] for i in idx], dtype=object)


def cluster_summary(clusters: pd.DataFrame, floors=SIZE_EDGES) -> pd.DataFrame:
    rows = []
    for (sym, gap), c in clusters.groupby(["symbol", "gap_sec"], sort=True):
        ok = c[(c["segment"] == "scheduled") & (~c["censored"])]
        basis = "notional_spot" if c["notional_spot"].notna().any() else "quote_raw"
        r = {"symbol": sym, "gap_sec": gap, "clusters_all": int(len(c)),
             "scheduled_uncensored": int(len(ok)),
             "censored": int(((c["segment"] == "scheduled") & c["censored"]).sum()),
             "manual": int((c["segment"] == "manual").sum()), "size_basis": basis}
        if len(ok):
            r.update({
                "n_p50": float(ok["n_events"].median()),
                "n_p90": float(ok["n_events"].quantile(0.9)),
                "n_p99": float(ok["n_events"].quantile(0.99)),
                "n_max": int(ok["n_events"].max()),
                "usd_p50": float(ok["size_usd"].median()),
                "usd_p90": float(ok["size_usd"].quantile(0.9)),
                "usd_p99": float(ok["size_usd"].quantile(0.99)),
                "usd_max": float(ok["size_usd"].max()),
                "dur_p90_sec": float(ok["duration_sec"].quantile(0.9)),
                "dur_max_sec": float(ok["duration_sec"].max())})
        for fl in floors:
            r[f"n_ge_{int(fl):d}"] = int((ok["size_usd"] >= fl).sum())
        rows.append(r)
    return pd.DataFrame(rows)


# ===================================================== bankruptcy price =====

def bankruptcy_bias(liq: pd.DataFrame, books: dict, tol_ms: int = 5000,
                    tight_ms: int = 1000) -> pd.DataFrame:
    """Bybit `price` vs Binance SPOT mid at the same instant, by side.

    Establishes the direction of the bankruptcy-price bias rather than
    asserting it. The raw comparison also contains the Bybit-perp vs
    Binance-spot basis plus timing noise, which carry the SAME sign for both
    sides, whereas a bankruptcy offset has OPPOSITE signs for longs and
    shorts. So half the between-side difference isolates the offset, and
    the mean of the two sides estimates the common component.
    """
    rows = []
    for asset, book in books.items():
        d = liq[liq["asset"] == asset].sort_values("timestamp")
        b = (book[["timestamp", "mid_price"]].dropna().sort_values("timestamp")
             .rename(columns={"timestamp": "book_ts"}))
        m = pd.merge_asof(d, b, left_on="timestamp", right_on="book_ts",
                          direction="nearest", tolerance=tol_ms)
        m = m.dropna(subset=["mid_price"])
        m["dt_ms"] = (m["timestamp"] - m["book_ts"]).abs()
        m["bias_bps"] = (m["price"] / m["mid_price"] - 1.0) * 1e4
        for side, s in m.groupby(m["side"].str.upper()):
            for label, ss in (("all", s), (f"dt<={tight_ms}ms", s[s["dt_ms"] <= tight_ms])):
                if not len(ss):
                    continue
                rows.append({
                    "asset": asset, "side": side,
                    "position_liquidated": "LONG" if side == "SELL" else "SHORT",
                    "subset": label, "n": int(len(ss)),
                    "median_bps": float(ss["bias_bps"].median()),
                    "mean_bps": float(ss["bias_bps"].mean()),
                    "p25_bps": float(ss["bias_bps"].quantile(0.25)),
                    "p75_bps": float(ss["bias_bps"].quantile(0.75)),
                    "frac_below_mid": float((ss["bias_bps"] < 0).mean()),
                    "median_dt_ms": float(ss["dt_ms"].median())})
    df = pd.DataFrame(rows)
    if len(df):
        for (_a, _s), g in df.groupby(["asset", "subset"]):
            if set(g["side"]) >= {"BUY", "SELL"}:
                buy = float(g.loc[g["side"] == "BUY", "median_bps"].iloc[0])
                sell = float(g.loc[g["side"] == "SELL", "median_bps"].iloc[0])
                df.loc[g.index, "bankruptcy_offset_bps"] = (buy - sell) / 2.0
                df.loc[g.index, "common_component_bps"] = (buy + sell) / 2.0
    return df


def book_cadence_in(book: pd.DataFrame, first_ms: int, last_ms: int) -> dict:
    """Snapshot cadence inside the labelled window, incl. the book collector's
    own hourly handoff (it runs on the same 3550s-per-hour pattern)."""
    b = book[(book["timestamp"] >= first_ms) & (book["timestamp"] <= last_ms)]
    if len(b) < 2:
        return {"n": int(len(b))}
    ts = b["timestamp"].values.astype("int64")
    gaps = np.diff(ts) / 1000.0
    big = gaps > 30
    soh = (ts[:-1][big] // 1000) % 3600
    return {"n": int(len(b)), "median_gap_sec": float(np.median(gaps)),
            "p99_gap_sec": float(np.quantile(gaps, 0.99)),
            "max_gap_sec": float(gaps.max()), "gaps_over_30s": int(big.sum()),
            "gaps_over_30s_starting_in_minute_59": int((soh >= 3540).sum()),
            "hours_in_gaps_over_30s": float(gaps[big].sum() / 3600.0)}
