"""
engines/forced_trade/cascade.py

T1 -- Scenario A2 (liquidation cascades): are they identifiable WITHOUT a
liquidation feed?

THE HYPOTHESIS
--------------
A margin engine force-closing underwater positions sends market orders. It does
not post bids. So a cascade should appear as a burst of one-sided AGGRESSIVE
flow, concurrent with a price move in the same direction, while resting depth is
consumed.

We can measure aggressor side exactly rather than inferring it: `trades` carries
`is_buyer_maker`, and (verified over the FULL span by validate_side_convention)
side="buy" <=> is_buyer_maker=0. A trade with is_buyer_maker=0 had a TAKER
BUYER; is_buyer_maker=1 had a TAKER SELLER. The `side` column is therefore the
aggressor side, and is fully redundant with `is_buyer_maker`.

THE DETECTOR (every term a parameter -- see CascadeParams)
    candidate at window t IF
        dominant_side_quote(t) > K * trailing_median(dominant_side_quote)
    AND abs(signed_imbalance(t))                 > I
    AND abs(price_move(t)) in trailing-vol units > M

    Adjacent candidates are merged into one EVENT (gap <= merge_gap_windows).

WHAT THIS DETECTOR CANNOT DO
----------------------------
It cannot distinguish a forced liquidation from a large discretionary market
order: both look identical in the tape. Without a liquidation feed there is no
label, so there is no true false-positive RATE to compute -- only corroborating
evidence and concentration statistics. Everything this module reports about
false positives is circumstantial by construction and is labelled as such
rather than dressed up as validation.

CAUSALITY
---------
Every trailing statistic is rolling(L).stat().shift(1): the window ending at
t-1, never including t. A detector that normalised by a statistic containing its
own observation would fire LESS often on exactly the largest events.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from engines.forced_trade.common import (
    DEFAULT_CACHE_DIR, DEFAULT_DB, MS_PER_SEC,
    days_between, ensure_dir, per_year, read_only_db,
)

logger = logging.getLogger("forced_trade.cascade")


# ============================================================== parameters ==

@dataclass(frozen=True)
class CascadeParams:
    """Detector settings. NOTHING here is hard-coded downstream.

    Defaults are the "base" setting of the sensitivity sweep. They were fixed
    before any output was inspected and are NOT tuned to a target event count.
    """
    window_sec: int = 60          # W: detection window
    burst_k: float = 5.0          # K: dominant-side volume vs trailing median
    imbalance_i: float = 0.80     # I: abs signed imbalance floor (0..1)
    move_m: float = 3.0           # M: abs price move in trailing-vol units
    lookback_windows: int = 1440  # L: trailing-statistic window (1 day at W=60s)
    merge_gap_windows: int = 1    # candidates this close merge into one event
    min_trailing_obs: int = 0     # extra warm-up beyond L before firing is allowed

    def label(self) -> str:
        return (f"W={self.window_sec}s K={self.burst_k:g} "
                f"I={self.imbalance_i:g} M={self.move_m:g} L={self.lookback_windows}")


# The sweep reported in the retro: three settings spanning loose -> strict so
# the SPREAD is visible. This is not a search; the spread IS the finding.
DEFAULT_SWEEP = (
    ("loose",  CascadeParams(burst_k=3.0,  imbalance_i=0.60, move_m=2.0)),
    ("base",   CascadeParams(burst_k=5.0,  imbalance_i=0.80, move_m=3.0)),
    ("strict", CascadeParams(burst_k=10.0, imbalance_i=0.90, move_m=4.0)),
)


def sweep_params(name: str = "base") -> CascadeParams:
    """The named sweep setting.

    Single source of truth. The window sweep and the T5 occupancy census both
    need "the base setting"; hard-coding K/I/M at each use site would let them
    drift apart, and an occupancy table computed from different thresholds than
    the count table it is supposed to explain would be silently wrong.
    """
    for n, p in DEFAULT_SWEEP:
        if n == name:
            return p
    raise KeyError(f"unknown sweep setting {name!r}; "
                   f"have {[n for n, _ in DEFAULT_SWEEP]}")


@dataclass
class CascadeResult:
    """Everything one (asset, setting) detector run produced."""
    asset: str
    setting: str
    params: CascadeParams
    events: pd.DataFrame
    n_windows: int
    n_candidate_windows: int
    span_days: float
    first_ms: int
    last_ms: int

    @property
    def n_events(self) -> int:
        return len(self.events)

    @property
    def events_per_year(self) -> float:
        return per_year(self.n_events, self.span_days)


# =================================================== bucket cache (1 scan) ==

_BUCKET_SQL = """
SELECT timestamp / :bucket_ms                                         AS bucket,
       COUNT(*)                                                       AS n_trades,
       SUM(CASE WHEN is_buyer_maker = 0 THEN quote_amount ELSE 0 END)  AS buy_quote,
       SUM(CASE WHEN is_buyer_maker = 1 THEN quote_amount ELSE 0 END)  AS sell_quote,
       MAX(timestamp)                                                 AS last_ts,
       price                                                          AS close_price
FROM trades
WHERE asset = :asset
GROUP BY bucket
ORDER BY bucket
"""
# NOTE ON THE BARE `price` COLUMN: SQLite documents that in an aggregate query
# containing EXACTLY ONE min()/max() aggregate, bare columns take their value
# from the row that produced it. MAX(timestamp) is the only such aggregate here,
# so close_price is the price of the LAST trade in the bucket. That is a
# documented guarantee, not an accident of query planning -- but --validate
# re-derives a random sample independently and asserts equality anyway.


def bucket_cache_path(asset, bucket_sec, cache_dir=DEFAULT_CACHE_DIR) -> Path:
    return Path(cache_dir) / f"trade_buckets_{asset}_{bucket_sec}s.parquet"


def build_bucket_cache(asset, bucket_sec=10, db_path=DEFAULT_DB,
                       cache_dir=DEFAULT_CACHE_DIR, force=False) -> pd.DataFrame:
    """Aggregate the full `trades` history for `asset` into fixed buckets.

    ONE full scan of a ~90M-row table per asset (minutes). Cached to parquet
    because the sensitivity sweep evaluates many settings against it. The cache
    holds RAW AGGREGATES ONLY -- no threshold is applied while building it, so
    it cannot influence any reported count.
    """
    path = bucket_cache_path(asset, bucket_sec, cache_dir)
    if path.exists() and not force:
        logger.info("[%s] bucket cache hit: %s", asset, path)
        return pd.read_parquet(path)

    ensure_dir(Path(cache_dir))
    bucket_ms = bucket_sec * MS_PER_SEC
    logger.info("[%s] building %ds bucket cache from trades "
                "(full scan, expect minutes) ...", asset, bucket_sec)
    t0 = time.time()
    with read_only_db(db_path) as conn:
        df = pd.read_sql_query(_BUCKET_SQL, conn,
                               params={"bucket_ms": bucket_ms, "asset": asset})
    logger.info("[%s] scan done: %d buckets in %.0fs", asset, len(df), time.time() - t0)

    df["bucket_start_ms"] = df["bucket"].astype("int64") * bucket_ms
    df = df[["bucket_start_ms", "n_trades", "buy_quote", "sell_quote",
             "last_ts", "close_price"]]
    df.to_parquet(path, index=False)
    logger.info("[%s] cached -> %s", asset, path)
    return df


def validate_bucket_cache(asset, df, bucket_sec, n_sample=25, seed=61,
                          db_path=DEFAULT_DB) -> dict:
    """Independently re-derive a random sample of cached buckets from source.

    Checks the two things the fast path assumes: (a) the SQLite bare-column rule
    really returned the LAST trade price, and (b) the buy/sell split matches a
    straightforward per-row recomputation.
    """
    rng = np.random.default_rng(seed)
    bucket_ms = bucket_sec * MS_PER_SEC
    idx = rng.choice(len(df), size=min(n_sample, len(df)), replace=False)
    bad_price, bad_flow, checked = [], [], 0

    with read_only_db(db_path) as conn:
        for i in idx:
            row = df.iloc[int(i)]
            t0 = int(row["bucket_start_ms"])
            raw = pd.read_sql_query(
                "SELECT timestamp, price, quote_amount, is_buyer_maker FROM trades "
                "WHERE asset=? AND timestamp >= ? AND timestamp < ?",
                conn, params=(asset, t0, t0 + bucket_ms))
            if raw.empty:
                continue
            checked += 1
            last = raw.loc[raw["timestamp"].idxmax()]
            if not np.isclose(float(last["price"]), float(row["close_price"])):
                bad_price.append((t0, float(last["price"]), float(row["close_price"])))
            buy = float(raw.loc[raw["is_buyer_maker"] == 0, "quote_amount"].sum())
            sell = float(raw.loc[raw["is_buyer_maker"] == 1, "quote_amount"].sum())
            if not (np.isclose(buy, float(row["buy_quote"]), rtol=1e-9)
                    and np.isclose(sell, float(row["sell_quote"]), rtol=1e-9)):
                bad_flow.append((t0, buy, float(row["buy_quote"])))

    ok = (not bad_price) and (not bad_flow)
    logger.info("[%s] cache validation: %d buckets re-derived, price_mismatch=%d "
                "flow_mismatch=%d -> %s", asset, checked, len(bad_price),
                len(bad_flow), "PASS" if ok else "FAIL")
    return {"asset": asset, "checked": checked, "price_mismatch": len(bad_price),
            "flow_mismatch": len(bad_flow), "ok": ok,
            "examples": (bad_price + bad_flow)[:5]}


def validate_side_convention(db_path=DEFAULT_DB, assets=("BTC", "ETH"),
                             full_scan=False, sample_days=14) -> dict:
    """Confirm side <-> is_buyer_maker. If this is wrong, every flow number in
    this module is wrong by the SIGN of the imbalance, so it is checked rather
    than assumed.

    TWO MODES, because the exhaustive one is not free:

      full_scan=False (default) -- count violations inside `sample_days` windows
        sampled from the start, middle and end of the span. These are index
        range-scans on (asset, timestamp) and cost seconds.

      full_scan=True -- count violations across every row.

    NOTE ON THE QUERY SHAPE: the obvious `GROUP BY side, is_buyer_maker` has no
    covering index, so SQLite builds a temporary B-tree over ~90M rows per asset
    and the check takes tens of minutes. Counting VIOLATIONS instead needs no
    grouping and no sort -- one streaming pass, same guarantee, and it reports
    the number that actually matters (how many rows break the rule) rather than
    a contingency table that has to be read to find out.
    """
    viol_sql = ("SELECT COUNT(*) FROM trades WHERE asset=? "
                "AND ((side='buy') <> (is_buyer_maker=0))")
    out = {}
    with read_only_db(db_path) as conn:
        for a in assets:
            t0 = time.time()
            if full_scan:
                viol = conn.execute(viol_sql, (a,)).fetchone()[0]
                total = conn.execute(
                    "SELECT COUNT(*) FROM trades WHERE asset=?", (a,)).fetchone()[0]
                scope = "FULL HISTORY"
                windows = None
            else:
                mn = conn.execute("SELECT MIN(timestamp) FROM trades WHERE asset=?",
                                  (a,)).fetchone()[0]
                mx = conn.execute("SELECT MAX(timestamp) FROM trades WHERE asset=?",
                                  (a,)).fetchone()[0]
                w = sample_days * 86400 * 1000 // 3
                mid = (mn + mx) // 2
                windows = [(mn, mn + w), (mid, mid + w), (max(mn, mx - w), mx)]
                viol = total = 0
                for lo, hi in windows:
                    viol += conn.execute(
                        viol_sql + " AND timestamp BETWEEN ? AND ?",
                        (a, lo, hi)).fetchone()[0]
                    total += conn.execute(
                        "SELECT COUNT(*) FROM trades WHERE asset=? AND timestamp "
                        "BETWEEN ? AND ?", (a, lo, hi)).fetchone()[0]
                scope = f"SAMPLED ({sample_days}d over 3 windows: start/mid/end)"
            out[a] = {"scope": scope, "rows_checked": total, "violations": viol,
                      "consistent": viol == 0, "elapsed_sec": round(time.time() - t0, 1),
                      "windows_ms": windows}
            logger.info("[%s] side/is_buyer_maker %s: %d violations in %s rows "
                        "-> %s (%.0fs)", a, scope, viol, f"{total:,}",
                        "PASS" if viol == 0 else "FAIL", time.time() - t0)
    return out


# ==================================================== window construction ==

def to_windows(buckets, bucket_sec, window_sec) -> pd.DataFrame:
    """Re-aggregate base buckets up to the detection window W.

    Gaps (buckets with no trades at all) are materialised with zero volume and a
    forward-filled close, so a quiet stretch cannot masquerade as missing data
    and silently shorten a trailing window.
    """
    if window_sec % bucket_sec != 0:
        raise ValueError(f"window_sec ({window_sec}) must be a multiple of "
                         f"bucket_sec ({bucket_sec})")
    win_ms = window_sec * MS_PER_SEC
    b = buckets.copy()
    b["win"] = (b["bucket_start_ms"] // win_ms).astype("int64")

    agg = b.groupby("win").agg(
        n_trades=("n_trades", "sum"),
        buy_quote=("buy_quote", "sum"),
        sell_quote=("sell_quote", "sum"),
        last_ts=("last_ts", "max"),
    )
    agg["close"] = b.sort_values("bucket_start_ms").groupby("win")["close_price"].last()

    full = pd.RangeIndex(int(agg.index.min()), int(agg.index.max()) + 1)
    agg = agg.reindex(full)
    n_gap = int(agg["n_trades"].isna().sum())
    for c in ("n_trades", "buy_quote", "sell_quote"):
        agg[c] = agg[c].fillna(0.0)
    agg["close"] = agg["close"].ffill()
    agg.index.name = "win"
    agg["ts_ms"] = agg.index.astype("int64") * win_ms
    if n_gap:
        logger.debug("to_windows W=%ds: %d empty windows materialised (%.4f%%)",
                     window_sec, n_gap, 100.0 * n_gap / len(agg))
    return agg.reset_index(drop=True)


# =============================================================== detector ==

def detect(windows, p: CascadeParams) -> pd.DataFrame:
    """Flag candidate windows. Adds diagnostic columns; applies no merging."""
    d = windows.copy()
    d["tot_quote"] = d["buy_quote"] + d["sell_quote"]
    d["dom_quote"] = d[["buy_quote", "sell_quote"]].max(axis=1)
    d["dom_side"] = np.where(d["buy_quote"] >= d["sell_quote"], 1, -1)
    d["imbalance"] = np.where(
        d["tot_quote"] > 0,
        (d["buy_quote"] - d["sell_quote"]) / d["tot_quote"].replace(0, np.nan),
        0.0)

    L = p.lookback_windows
    # STRICTLY TRAILING: rolling window ending at t-1 (shift AFTER rolling).
    d["med_dom"] = d["dom_quote"].rolling(L, min_periods=L).median().shift(1)
    d["ret"] = d["close"].pct_change()
    d["sd_ret"] = d["ret"].rolling(L, min_periods=L).std().shift(1)

    d["burst_ratio"] = np.where(d["med_dom"] > 0, d["dom_quote"] / d["med_dom"], np.nan)
    d["move_z"] = np.where(d["sd_ret"] > 0, d["ret"] / d["sd_ret"], np.nan)

    warm = L + p.min_trailing_obs
    ready = pd.Series(d.index >= warm, index=d.index)

    d["cand"] = (
        ready
        & (d["burst_ratio"] > p.burst_k)
        & (d["imbalance"].abs() > p.imbalance_i)
        & (d["move_z"].abs() > p.move_m)
    ).fillna(False)
    # Diagnostic ONLY -- not part of the firing rule. A genuine cascade should
    # push price the way the aggressive flow points; a plain volume burst need
    # not. Reported as corroboration, never used to filter.
    d["dir_agree"] = np.sign(d["imbalance"]) == np.sign(d["ret"])
    return d


def merge_events(d, p: CascadeParams, window_sec) -> pd.DataFrame:
    """Collapse runs of candidate windows into events."""
    cols = ["start_ms", "end_ms", "n_windows", "duration_sec", "dom_side",
            "peak_burst", "mean_imbalance", "max_abs_move_z", "event_ret",
            "quote_volume", "dir_agree_frac"]
    cand_idx = np.flatnonzero(d["cand"].values)
    if cand_idx.size == 0:
        return pd.DataFrame(columns=cols)

    groups, cur = [], [cand_idx[0]]
    for i in cand_idx[1:]:
        if i - cur[-1] <= p.merge_gap_windows:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)

    win_ms = window_sec * MS_PER_SEC
    rows = []
    for g in groups:
        lo, hi = int(g[0]), int(g[-1])
        seg = d.iloc[lo:hi + 1]
        c0 = d["close"].iloc[lo - 1] if lo > 0 else d["close"].iloc[lo]
        c1 = d["close"].iloc[hi]
        rows.append({
            "start_ms": int(d["ts_ms"].iloc[lo]),
            "end_ms": int(d["ts_ms"].iloc[hi]) + win_ms,
            "n_windows": hi - lo + 1,
            "duration_sec": (hi - lo + 1) * window_sec,
            "dom_side": int(seg.loc[seg["dom_quote"].idxmax(), "dom_side"]),
            "peak_burst": float(seg["burst_ratio"].max()),
            "mean_imbalance": float(seg["imbalance"].mean()),
            "max_abs_move_z": float(seg["move_z"].abs().max()),
            "event_ret": float(c1 / c0 - 1.0) if c0 else np.nan,
            "quote_volume": float(seg["tot_quote"].sum()),
            "dir_agree_frac": float(seg["dir_agree"].mean()),
        })
    return pd.DataFrame(rows, columns=cols)


def run_detector(asset, buckets, bucket_sec, setting, p: CascadeParams) -> CascadeResult:
    w = to_windows(buckets, bucket_sec, p.window_sec)
    d = detect(w, p)
    ev = merge_events(d, p, p.window_sec)
    first_ms, last_ms = int(w["ts_ms"].iloc[0]), int(w["ts_ms"].iloc[-1])
    res = CascadeResult(
        asset=asset, setting=setting, params=p, events=ev,
        n_windows=len(w), n_candidate_windows=int(d["cand"].sum()),
        span_days=days_between(first_ms, last_ms),
        first_ms=first_ms, last_ms=last_ms)
    logger.info("[%s/%s] %s -> %d candidate windows, %d merged events "
                "(%.1f/yr over %.1f days)", asset, setting, p.label(),
                res.n_candidate_windows, res.n_events, res.events_per_year,
                res.span_days)
    return res
