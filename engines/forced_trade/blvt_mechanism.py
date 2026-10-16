"""
engines/forced_trade/blvt_mechanism.py

Cycle 62B T1 -- establish the leveraged-token rebalance mechanism FROM THE DATA.

Binance has taken its BLVT documentation down (the FAQ pages return "This article
does not exist") and the historical NAV endpoint `fapi/v1/lvtKlines` now returns
404. What survives is the token's MARKET price (spot 1m klines) and the
underlying perp (USDT-M 1m klines). This module shows that is enough to locate
rebalances and to read the trigger off them.

WHAT IS CONSTANT BETWEEN REBALANCES
-----------------------------------
A leveraged token holds a basket of perpetual contracts. Per token, that basket
is a fixed quantity q of the perp between rebalances, so

    NAV_t = a + q * F_t      (F = perp price; funding and fees accrue to cash)

LINEAR IN PRICE LEVELS, and the real leverage L = q F / NAV drifts mechanically
with price. Only a rebalance changes q, so a rebalance is a change in the slope
of token price on perp price. q is per token, so a redenomination also moves
it -- but it moves the token price by the same factor and leaves L unchanged,
which is how it is told apart (L_pre ~= L_post).

WHY LEVELS, NOT MINUTE RETURNS
------------------------------
NAV is not available; market price = NAV + premium. The premium is bounded and
mean-reverting, so it swamps a one-minute price change but not a twelve-hour
price range. A slope fitted to minute differences over an hour was too noisy
to tell a rebalance from nothing (the synthetic null produced 40 false events
in 30 days); a slope fitted to levels over twelve hours is not. The cost is
resolution: two rebalances closer than `window_h` merge into one event.

THE TIMING TEST NEEDS NO THRESHOLD
----------------------------------
`boundary_profile` fits short windows either side of every UTC hour boundary and
averages |log q_after/q_before| by hour of day. A scheduled reset puts a spike
at its hour; a trigger does not. Positive control: the FTX BULL/BEAR tokens
Binance listed in 2020-Q1 were, by FTX's design, reset once a day at a fixed
time. If the detector cannot see that, its silence on BLVTs means nothing.

SCOPE GUARD: this module LOCATES rebalances and characterises the TRIGGER
(leverage before/after, time of day, move since the previous rebalance). It does
not measure price behaviour around them -- that is T3, gated behind T2.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from engines.xsec_reversal.archive import ArchiveClient, month_range

logger = logging.getLogger("forced_trade.blvt_mechanism")

DEFAULT_CACHE = Path("data/external/binance_archive")
DEFAULT_OUT = Path("outputs/forced_trade/c62b")


@dataclass(frozen=True)
class DetectParams:
    """Every knob, stated up front. Sensitivity is reported, not tuned."""
    window_h: int = 12          # hours each side for the before/after level fits
    profile_window_h: int = 3   # shorter windows for the hour-of-day profile
    jump_log: float = 0.10      # |log q_after/q_before| that counts as a rebalance
    min_cover: float = 0.5      # share of minutes with a token print in each window
    min_rel_sd: float = 0.002   # the perp must move: sd(F)/mean(F) in each window
    min_seg_min: int = 60       # minutes each side of a split, when placing it
    hac_lags: int = 120         # Newey-West lags (minutes): the premium is persistent
    min_z: float = 4.0          # |jump| / HAC SE for an event to count as significant


# ------------------------------------------------------------------ data ---

def _vwap(k: pd.DataFrame) -> pd.Series:
    """Per-minute VWAP; NaN where nothing traded. A no-trade minute's close is
    stale (31.5% of BTCUP minutes), which is noise the slope does not need."""
    with np.errstate(divide="ignore", invalid="ignore"):
        v = k["quote_asset_volume"] / k["volume"]
    return v.where(k["volume"] > 0)


def load_minutes(token: str, perp: str, start: str, end: str,
                 cache_dir: Path = DEFAULT_CACHE, price: str = "vwap") -> pd.DataFrame:
    """Aligned 1m prices: token market price P (spot) and perp price F."""
    periods = month_range(start, end)
    spot = ArchiveClient(cache_dir=cache_dir, market="spot")
    fut = ArchiveClient(cache_dir=cache_dir, market="futures/um")
    tok = spot.load_symbol_range(token, "1m", periods, use_listing=False)
    per = fut.load_symbol_range(perp, "1m", periods, use_listing=False)
    if tok is None or per is None:
        raise RuntimeError(f"missing 1m data for {token} or {perp}")
    pick = _vwap if price == "vwap" else (lambda k: k["close"])
    df = pd.DataFrame({"P": pick(tok), "tok_trades": tok["num_trades"],
                       "F": pick(per)}).dropna(subset=["P", "F"])
    df = df[(df["P"] > 0) & (df["F"] > 0)]
    logger.info("[%s/%s] %d aligned minutes (%s) %s -> %s", token, perp, len(df),
                price, df.index.min(), df.index.max())
    return df


class _Levels:
    """Cumulative sums on a regular 1m grid, so the OLS of P on F over ANY
    window [i, j) is O(1). Prices are centred on their global means first --
    the same constant everywhere, so every window fit is exact."""

    def __init__(self, df: pd.DataFrame):
        start = df.index.min().floor("h")
        grid = pd.date_range(start, df.index.max(), freq="1min")
        full = df[["P", "F"]].reindex(grid)
        self.index = grid
        self.F = full["F"].to_numpy(float)
        self.P = full["P"].to_numpy(float)
        w = np.isfinite(self.F) & np.isfinite(self.P)
        self.F0 = float(np.nanmean(self.F))
        self.P0 = float(np.nanmean(self.P))
        f = np.where(w, self.F - self.F0, 0.0)
        p = np.where(w, self.P - self.P0, 0.0)
        z = np.zeros(1)
        self.cs = {k: np.concatenate([z, np.cumsum(v)]) for k, v in
                   {"n": w.astype(float), "f": f, "p": p, "ff": f * f,
                    "fp": f * p, "pp": p * p}.items()}

    def fit(self, i, j):
        """OLS P = a + q F on minutes [i, j). Arrays welcome.

        Returns q, n (minutes present), rel_sd (sd(F)/mean(F)), sse.
        """
        c = self.cs
        n = c["n"][j] - c["n"][i]
        sf, sp = c["f"][j] - c["f"][i], c["p"][j] - c["p"][i]
        sff, sfp = c["ff"][j] - c["ff"][i], c["fp"][j] - c["fp"][i]
        spp = c["pp"][j] - c["pp"][i]
        with np.errstate(divide="ignore", invalid="ignore"):
            sxx = sff - sf * sf / n
            sxy = sfp - sf * sp / n
            syy = spp - sp * sp / n
            q = sxy / sxx
            sse = syy - sxy * sxy / sxx
            rel_sd = np.sqrt(np.maximum(sxx, 0) / n) / (sf / n + self.F0)
        return q, n, rel_sd, sse


# -------------------------------------------------------- window slopes ---

def window_slopes(df: pd.DataFrame, window_h: int,
                  p: DetectParams = DetectParams(), lv: _Levels = None) -> pd.DataFrame:
    """At every UTC hour boundary: q fitted on the `window_h` hours before it
    and on the `window_h` hours after it, d = log(q_after / q_before)."""
    lv = lv or _Levels(df)
    W = window_h * 60
    N = len(lv.index)
    k = np.arange(W, N - W + 1, 60)
    k = k[lv.index[np.minimum(k, N - 1)].minute == 0]
    qb, nb, sb, _ = lv.fit(k - W, k)
    qa, na, sa, _ = lv.fit(k, k + W)
    ok = ((nb >= p.min_cover * W) & (na >= p.min_cover * W)
          & (sb >= p.min_rel_sd) & (sa >= p.min_rel_sd)
          & (np.sign(qb) == np.sign(qa)) & (qb != 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(ok, np.log(qa / qb), np.nan)
    Fk = lv.F[k]
    Pk = lv.P[k]
    out = pd.DataFrame({"q_before": qb, "q_after": qa, "d": d,
                        "L_before": qb * Fk / Pk, "L_after": qa * Fk / Pk,
                        "rel_sd_before": sb, "rel_sd_after": sa},
                       index=lv.index[k])
    out.index.name = "boundary"
    return out


def boundary_profile(df: pd.DataFrame, sign: int, p: DetectParams = DetectParams(),
                     lv: _Levels = None) -> pd.DataFrame:
    """Threshold-free timing test: |d| across each UTC hour boundary, by hour."""
    ws = window_slopes(df, p.profile_window_h, p, lv)
    ws = ws[np.sign(ws["q_before"]) == sign]
    a = ws["d"].abs().dropna()
    by = a.groupby(a.index.hour)
    prof = pd.DataFrame({"n": by.size(), "mean_abs_d": by.mean(),
                         "median_abs_d": by.median(), "p90_abs_d": by.quantile(0.9)})
    # Each hour against all the others: a scheduled hour stands out by many SE.
    z = {}
    for h in prof.index:
        x, rest = a[a.index.hour == h], a[a.index.hour != h]
        se = np.sqrt(x.var() / len(x) + rest.var() / len(rest))
        z[h] = (x.mean() - rest.mean()) / se if se > 0 else np.nan
    prof["z_vs_rest"] = pd.Series(z)
    prof.index.name = "utc_hour"
    return prof


def schedule_profile(df: pd.DataFrame, sign: int, p: DetectParams = DetectParams(),
                     lv: _Levels = None) -> pd.DataFrame:
    """The powerful timing test: does q reset at a fixed hour BY THE AMOUNT a
    reset must?

    A scheduled reset at hour H undoes the leverage drift since the previous
    reset, so its size is predictable from the prior day's move r:
        long token (L > 1):   d(H) ~ (L - 1) * r
        short token:          d(H) ~ -(|L| + 1) * r
    For each UTC hour, regress d on the perp's log move over the preceding 24 h
    (through the origin, HC1 SE). A schedule shows as one hour with a large,
    correctly signed beta; a trigger leaves every hour near zero. The mean-|d|
    profile cannot see small daily resets under premium noise (a synthetic
    year at 2x showed z ~0-2 at the planted hour); this regression can, because
    it uses the predicted sign and size.
    """
    lv = lv or _Levels(df)
    ws = window_slopes(df, p.profile_window_h, p, lv)
    ws = ws[np.sign(ws["q_before"]) == sign]
    Fs = pd.Series(lv.F, index=lv.index).ffill()
    f_now = Fs.reindex(ws.index).to_numpy()
    f_prev = Fs.reindex(ws.index - pd.Timedelta(hours=24)).to_numpy()
    x = pd.Series(np.log(f_now / f_prev), index=ws.index)
    d = ws["d"]
    ok = x.notna() & d.notna()
    x, d = x[ok], d[ok]
    rows = []
    for h in range(24):
        xs, ds = x[x.index.hour == h].to_numpy(), d[d.index.hour == h].to_numpy()
        n = len(xs)
        if n < 10:
            rows.append({"utc_hour": h, "n": n})
            continue
        sxx = float(xs @ xs)
        beta = float(xs @ ds) / sxx
        e = ds - beta * xs
        se = float(np.sqrt((xs * xs * e * e).sum() * n / (n - 1)) / sxx)
        rows.append({"utc_hour": h, "n": n, "beta": beta, "se": se, "z": beta / se})
    return pd.DataFrame(rows).set_index("utc_hour")


# ------------------------------------------------------------- detection ---

def detect_events(ws: pd.DataFrame, sign: int, p: DetectParams = DetectParams()) -> pd.DataFrame:
    """Boundaries where |d| >= jump_log and is the largest within +-window_h."""
    s = ws[np.sign(ws["q_before"]) == sign]["d"].asfreq("h")
    a = s.abs()
    peak = a.rolling(2 * p.window_h + 1, center=True, min_periods=1).max()
    hit = (a >= p.jump_log) & (a == peak)
    return pd.DataFrame({"boundary": s.index[hit], "d_window": s[hit].to_numpy()})


def _hac_slope(lv: _Levels, i: int, j: int, lags: int):
    """Slope of P on F over minutes [i, j) with a Newey-West (Bartlett) SE.

    Lags are in clock minutes (missing minutes contribute zero), because the
    premium's persistence is a function of time, not of print count.
    """
    F, P = lv.F[i:j], lv.P[i:j]
    w = np.isfinite(F) & np.isfinite(P)
    if w.sum() < 30:
        return np.nan, np.nan
    x = np.where(w, F - F[w].mean(), 0.0)
    sxx = float((x * x).sum())
    q = float((x * np.where(w, P - P[w].mean(), 0.0)).sum() / sxx)
    e = np.where(w, P - P[w].mean() - q * x, 0.0)
    u = x * e
    K = min(lags, len(u) - 1)
    g = [float(u @ u)] + [float(u[k:] @ u[:-k]) for k in range(1, K + 1)]
    s = g[0] + 2 * sum((1 - k / (K + 1)) * g[k] for k in range(1, K + 1))
    return q, float(np.sqrt(max(s, 0.0)) / sxx)


def refine_minute(lv: _Levels, boundary: pd.Timestamp, p: DetectParams = DetectParams()) -> dict:
    """Place one rebalance to the minute: the split tau anywhere in
    [b - W, b + W) minimising SSE of separate level fits either side.

    The search spans the whole window, not a neighbourhood of the flagged
    boundary: a level fit is dominated by the part of its window where the
    price moved most, so |d| plateaus for hours around a rebalance and the
    flagged boundary can sit hours from the true one.
    """
    W, m = p.window_h * 60, p.min_seg_min
    k = int(lv.index.get_loc(boundary))
    lo, hi = k - W, k + W
    taus = np.arange(lo + m, hi - m + 1)
    if lo < 0 or hi > len(lv.index) or len(taus) == 0:
        return {}
    q1, n1, _, e1 = lv.fit(np.full_like(taus, lo), taus)
    q2, n2, _, e2 = lv.fit(taus, np.full_like(taus, hi))
    sse = np.where((n1 >= m // 2) & (n2 >= m // 2), e1 + e2, np.inf)
    if not np.isfinite(sse).any():
        return {}
    b = int(np.argmin(sse))
    t = int(taus[b])
    F, P = lv.F[t], lv.P[t]
    if not (np.isfinite(F) and np.isfinite(P)):
        j = t + np.nonzero(np.isfinite(lv.F[t:]) & np.isfinite(lv.P[t:]))[0][0]
        F, P = lv.F[j], lv.P[j]
    qa, sa = _hac_slope(lv, lo, t, p.hac_lags)
    qb, sb = _hac_slope(lv, t, hi, p.hac_lags)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.log(qb / qa) / np.sqrt((sa / qa) ** 2 + (sb / qb) ** 2)
    return {"time": lv.index[t], "q_pre": float(q1[b]), "q_post": float(q2[b]),
            "L_pre": float(q1[b] * F / P), "L_post": float(q2[b] * F / P),
            "F": float(F), "P": float(P), "z": float(z)}


def characterise(lv: _Levels, events: pd.DataFrame, p: DetectParams = DetectParams()) -> pd.DataFrame:
    """Minute time, leverage before/after, and the move since the last event."""
    rows = []
    for _, e in events.iterrows():
        r = refine_minute(lv, e["boundary"], p)
        if r:
            r["d_window"] = e["d_window"]
            rows.append(r)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Resolution limit: two rebalances closer than window_h are one event here
    # (the larger window jump is kept).
    out = out.sort_values("time").reset_index(drop=True)
    keep = []
    for i, r in out.iterrows():
        if keep and r["time"] - out.loc[keep[-1], "time"] < pd.Timedelta(hours=p.window_h):
            if abs(r["d_window"]) > abs(out.loc[keep[-1], "d_window"]):
                keep[-1] = i
            continue
        keep.append(i)
    out = out.loc[keep].reset_index(drop=True)
    out["utc_hour"] = out["time"].dt.hour
    out["utc_minute"] = out["time"].dt.minute
    out["dlogq"] = np.log(out["q_post"] / out["q_pre"])
    out["dlogL"] = np.log(out["L_post"] / out["L_pre"])
    # A redenomination moves q and the price together and leaves leverage alone.
    out["redenomination_like"] = (out["dlogq"].abs() > 0.5) & (out["dlogL"].abs() < 0.1)
    out["significant"] = ((out["dlogq"].abs() >= p.jump_log) & (out["z"].abs() >= p.min_z)
                          & ~out["redenomination_like"])
    # Intervals and moves are measured between SIGNIFICANT events only.
    sig = out[out["significant"]]
    out["days_since_prev"] = sig["time"].diff().dt.total_seconds() / 86400
    out["perp_logret_since_prev"] = np.log(sig["F"] / sig["F"].shift(1))
    out["direction"] = np.where(out["L_post"].abs() < out["L_pre"].abs(),
                                "delever", "relever")
    return out


# --------------------------------------------------------------- summary ---

def summarise(name: str, sign: int, lv: _Levels, ev: pd.DataFrame,
              prof: pd.DataFrame, band=(1.25, 4.0)) -> dict:
    present = np.isfinite(lv.F) & np.isfinite(lv.P)
    first, last = lv.index[present][0], lv.index[present][-1]
    # Rates use the CALENDAR span. Minutes with a print are fewer (no-trade
    # minutes are gaps under VWAP), and dividing by them inflated the rate.
    span_days = (last - first).total_seconds() / 86400
    s = {"token": name, "sign": sign, "first": str(first), "last": str(last),
         "calendar_days": round(float(span_days), 1),
         "minutes_with_both_prices_days": round(float(present.sum() / 1440), 1),
         "n_events": int(len(ev))}
    if len(prof):
        m = prof["mean_abs_d"]
        s["profile_peak_hour"] = int(m.idxmax())
        s["profile_peak_to_median"] = round(float(m.max() / m.median()), 2)
        s["profile_peak_z"] = round(float(prof["z_vs_rest"].max()), 2)
        s["profile_hours_z_ge_4"] = [int(h) for h in prof.index[prof["z_vs_rest"] >= 4]]
    if "sched_z" in prof:
        z = prof["sched_z"]
        top = int(z.abs().idxmax())
        # A reset undoes drift: long tokens beta > 0, short tokens beta < 0.
        right = np.sign(z) == sign
        s["sched_top_hour"] = top
        s["sched_top_z"] = round(float(z.loc[top]), 2)
        s["sched_top_beta"] = round(float(prof.loc[top, "sched_beta"]), 3)
        s["sched_hours_z_ge_4_right_sign"] = [int(h) for h in z.index[right & (z.abs() >= 4)]]
        s["sched_hours_z_ge_4_wrong_sign"] = [int(h) for h in z.index[~right & (z.abs() >= 4)]]
        # Concentrated (a schedule) or spread over the day (a trigger)?
        b, se = prof["sched_beta"], prof["sched_se"]
        w = 1 / se ** 2
        bw = float((w * b).sum() / w.sum())
        Q = float((((b - bw) / se) ** 2).sum())
        from scipy.stats import chi2 as _chi2
        s["sched_hours_right_sign"] = int(right.sum())
        s["sched_pooled_beta"] = round(bw, 3)
        s["sched_homogeneity_Q_df23"] = round(Q, 1)
        s["sched_homogeneity_p"] = float(f"{_chi2.sf(Q, len(b) - 1):.3g}")
        s["sched_top_to_median_beta"] = round(float(b.loc[top] / b.median()), 2)
    if len(ev) and ev["significant"].any():
        real = ev[ev["significant"]]
        hours = real["utc_hour"].value_counts().reindex(range(24), fill_value=0)
        exp = len(real) / 24
        chi2 = float(((hours - exp) ** 2 / exp).sum()) if exp > 0 else float("nan")
        Lpre = real["L_pre"].abs()
        # Sign test: a rebalance undoes the leverage drift the move since the
        # last one created. Long token: price up -> leverage down -> relever.
        # Short token: price up -> leverage up -> delever. Noise has no sign.
        mv = real["perp_logret_since_prev"]
        expect_relever = (mv > 0) if sign > 0 else (mv < 0)
        has = mv.notna() & (mv != 0)
        agree = ((real["direction"] == "relever") == expect_relever)[has]
        s.update({
            "n_rebalances": int(len(real)),
            "n_candidates_not_significant": int((~ev["significant"]).sum()),
            "sign_agreement_with_drift": round(float(agree.mean()), 3) if len(agree) else None,
            "n_redenomination_like": int(ev["redenomination_like"].sum()),
            "rebalances_per_year": round(len(real) / span_days * 365.25, 1),
            "median_days_between": round(float(real["days_since_prev"].median()), 2),
            "events_in_00utc_hour": int(hours.loc[0]),
            "max_events_in_any_hour": int(hours.max()),
            "hour_of_day_chi2_df23": round(chi2, 1),
            "L_pre_abs_q10_q50_q90": [round(float(x), 2) for x in Lpre.quantile([0.1, 0.5, 0.9])],
            "L_post_abs_q10_q50_q90": [round(float(x), 2) for x in real["L_post"].abs().quantile([0.1, 0.5, 0.9])],
            "share_L_pre_outside_band": round(float(((Lpre < band[0]) | (Lpre > band[1])).mean()), 3),
            "delever_share": round(float((real["direction"] == "delever").mean()), 3),
            "median_abs_perp_move_since_prev": round(float(real["perp_logret_since_prev"].abs().median()), 3),
        })
    return s


def scheduled_share(prof: pd.DataFrame, ws: pd.DataFrame, sign: int) -> dict:
    """How much of a full daily reset happens at each fixed hour.

    share_h = (beta_h - beta_rest) / beta_theory, with beta_theory = L - 1
    (long) or -(L + 1) (short) at the token's median leverage. A pure daily
    schedule gives ~1 at its hour and ~0 elsewhere; a pure trigger ~0 at all.
    """
    L = float(ws["L_before"].abs().median())
    bt = (L - 1) if sign > 0 else -(L + 1)
    b, se = prof["sched_beta"], prof["sched_se"]
    sh = {}
    for h in prof.index:
        w = 1 / se.drop(h) ** 2
        rest = float((w * b.drop(h)).sum() / w.sum())
        sh[int(h)] = ((b[h] - rest) / bt, float(np.sqrt(se[h] ** 2 + 1 / w.sum())) / abs(bt))
    top = max(sh, key=lambda h: sh[h][0])
    r = lambda v: round(float(v), 3)
    return {"L_median": r(L), "beta_theory": r(bt),
            "share_h00": r(sh[0][0]), "share_h00_se": r(sh[0][1]),
            "share_h01": r(sh[1][0]), "share_h01_se": r(sh[1][1]),
            "share_max_hour": top, "share_max": r(sh[top][0]), "share_max_se": r(sh[top][1])}


def detect(df: pd.DataFrame, sign: int, p: DetectParams = DetectParams()):
    """Full pass on aligned minutes: (levels, window slopes, events, profile)."""
    lv = _Levels(df)
    ws = window_slopes(df, p.window_h, p, lv)
    ev = characterise(lv, detect_events(ws, sign, p), p)
    prof = boundary_profile(df, sign, p, lv)
    prof = prof.join(schedule_profile(df, sign, p, lv)[["beta", "se", "z"]]
                     .rename(columns={"beta": "sched_beta", "se": "sched_se",
                                      "z": "sched_z"}))
    return lv, ws, ev, prof


def run_token(token: str, perp: str, start: str, end: str, sign: int,
              p: DetectParams = DetectParams(), cache_dir: Path = DEFAULT_CACHE,
              out_dir: Path = DEFAULT_OUT, tag: str = "", save: bool = True,
              save_slopes: bool = False):
    df = load_minutes(token, perp, start, end, cache_dir)
    lv, ws, ev, prof = detect(df, sign, p)
    summ = summarise(token, sign, lv, ev, prof)
    summ.update(scheduled_share(prof, ws, sign))
    summ["params"] = asdict(p)
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"t1_{token}{tag}"
        ev.to_csv(out_dir / f"{stem}_events.csv", index=False)
        prof.to_csv(out_dir / f"{stem}_boundary_profile.csv")
        if save_slopes:  # ~1.7 MB per multi-year token; regenerable by `batch`
            ws.to_csv(out_dir / f"{stem}_window_slopes.csv.gz", compression="gzip")
        (out_dir / f"{stem}_summary.json").write_text(
            json.dumps(summ, indent=2, default=str), encoding="utf-8")
    return df, ws, prof, ev, summ


# family, token, perp, first month, last month, sign. BLVT = Binance's own
# UP/DOWN tokens; FTX = the FTX-issued BULL/BEAR tokens Binance listed in
# 2020-Q1, reset daily by design -- the POSITIVE CONTROL for the timing tests.
T1_TOKENS = (
    ("BLVT", "BTCUPUSDT", "BTCUSDT", "2020-05", "2024-02", 1),
    ("BLVT", "BTCDOWNUSDT", "BTCUSDT", "2020-05", "2024-02", -1),
    ("BLVT", "ETHUPUSDT", "ETHUSDT", "2020-07", "2024-02", 1),
    ("BLVT", "ETHDOWNUSDT", "ETHUSDT", "2020-07", "2024-02", -1),
    ("BLVT", "BNBUPUSDT", "BNBUSDT", "2020-08", "2024-02", 1),
    ("BLVT", "BNBDOWNUSDT", "BNBUSDT", "2020-08", "2024-02", -1),
    ("BLVT", "ADAUPUSDT", "ADAUSDT", "2020-07", "2023-08", 1),
    ("BLVT", "ADADOWNUSDT", "ADAUSDT", "2020-07", "2023-08", -1),
    ("FTX", "BULLUSDT", "BTCUSDT", "2020-01", "2020-03", 1),
    ("FTX", "BEARUSDT", "BTCUSDT", "2020-01", "2020-03", -1),
    ("FTX", "ETHBULLUSDT", "ETHUSDT", "2020-01", "2020-03", 1),
    ("FTX", "ETHBEARUSDT", "ETHUSDT", "2020-01", "2020-03", -1),
)

TABLE_COLS = ("family", "token", "first", "last", "calendar_days", "n_rebalances",
              "rebalances_per_year", "median_days_between", "sign_agreement_with_drift",
              "L_median", "L_pre_abs_q10_q50_q90", "events_in_00utc_hour",
              "max_events_in_any_hour", "sched_hours_right_sign", "sched_homogeneity_Q_df23",
              "sched_homogeneity_p", "sched_top_hour", "sched_top_z",
              "sched_top_to_median_beta", "share_h00", "share_h00_se", "share_h01",
              "share_h01_se", "share_max_hour", "share_max", "share_max_se")


def t1_table(out_dir: Path = DEFAULT_OUT, tokens=T1_TOKENS) -> pd.DataFrame:
    rows = []
    for fam, tok, *_ in tokens:
        f = out_dir / f"t1_{tok}_summary.json"
        if not f.exists():
            logger.warning("no summary for %s -- run it first", tok)
            continue
        s = json.loads(f.read_text(encoding="utf-8"))
        s["family"] = fam
        rows.append({c: s.get(c) for c in TABLE_COLS})
    t = pd.DataFrame(rows)
    t.to_csv(out_dir / "t1_mechanism_table.csv", index=False)
    return t


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Cycle 62B T1: locate leveraged-token "
                                             "rebalances from 1m market prices")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="one token")
    r.add_argument("--token", required=True)
    r.add_argument("--perp", required=True)
    r.add_argument("--start", required=True)
    r.add_argument("--end", required=True)
    r.add_argument("--sign", type=int, choices=(1, -1), required=True,
                   help="+1 long token (UP/BULL), -1 short token (DOWN/BEAR)")
    r.add_argument("--tag", default="")
    r.add_argument("--no-save", action="store_true")
    sub.add_parser("batch", help="every token in T1_TOKENS, then the table")
    sub.add_parser("table", help="summary table from saved per-token outputs")
    for s in (r, sub.choices["batch"]):
        s.add_argument("--keep-slopes", action="store_true",
                       help="also write the hour-boundary slope series (large)")
        s.add_argument("--window-h", type=int, default=DetectParams.window_h)
        s.add_argument("--profile-window-h", type=int, default=DetectParams.profile_window_h)
        s.add_argument("--jump-log", type=float, default=DetectParams.jump_log)
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("-v", "--verbose", action="count", default=0)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    out = Path(a.out_dir)
    if a.cmd in ("batch", "table"):
        if a.cmd == "batch":
            p = DetectParams(window_h=a.window_h, profile_window_h=a.profile_window_h,
                             jump_log=a.jump_log)
            for fam, tok, perp, s0, s1, sign in T1_TOKENS:
                run_token(tok, perp, s0, s1, sign, p, Path(a.cache_dir), out,
                          save_slopes=a.keep_slopes)
                print(f"done {fam} {tok}", flush=True)
        t = t1_table(out)
        with pd.option_context("display.width", 250, "display.max_columns", 40):
            print(t.to_string(index=False))
        return 0
    p = DetectParams(window_h=a.window_h, profile_window_h=a.profile_window_h,
                     jump_log=a.jump_log)
    _, _, prof, ev, summ = run_token(a.token, a.perp, a.start, a.end, a.sign, p,
                                     Path(a.cache_dir), out, a.tag, save=not a.no_save,
                                     save_slopes=a.keep_slopes)
    print(json.dumps(summ, indent=2, default=str))
    print("\nboundary profile (|log q_after/q_before| across each UTC hour boundary, "
          f"{p.profile_window_h}h windows):")
    print(prof.round(4).to_string())
    if len(ev):
        cols = ["time", "L_pre", "L_post", "dlogq", "direction", "days_since_prev",
                "perp_logret_since_prev", "redenomination_like"]
        print("\nevents:")
        print(ev[cols].round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
