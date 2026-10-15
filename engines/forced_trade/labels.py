"""
engines/forced_trade/labels.py

Cycle 66 T1 -- the labelled liquidation data: input snapshot, coverage mask,
clusters.

WHY THIS EXISTS
---------------
Cycle 61 built an A2 flow-burst detector and could not compute a false-positive
rate because nothing in the tape distinguishes a forced liquidation from a large
discretionary market order. Cycle 62A started recording Bybit `allLiquidation`
on 2026-08-19. This module turns those rows into something a detector can be
scored against, without letting a collector outage read as a quiet market.

THE COVERAGE MASK IS BUILT FROM THE COLLECTOR LOG, NOT FROM collector_gaps
-------------------------------------------------------------------------
`collector_gaps` only records disconnects INSIDE a run. It cannot see:
  * the ~50s handoff every hour (each scheduled run lasts 3550s),
  * hourly runs that never started,
  * runs that were killed and never printed an exit line.
The log has one UTC `window :` line per completed run, the local-time
`connected to` line of every (re)connect and a UTC `GAP OPENED` line at every
disconnect. Observed time is therefore reconstructed from the log as
[connect, next disconnect or run end]. `collector_gaps` is kept as a
cross-check, with its retry chains merged (see merge_gap_chains).

THE MANUAL 62A SEGMENT
----------------------
The first rows (2026-08-19 15:27 UTC) were written by the manual Cycle 62A
verification runs, which started at or during a $180M cascade and are not in
the service log. That cascade may be scored as an EVENT. It must never feed a
baseline, control or chance rate computed from the liquidations table --
there is no pre-event observation from that session, so using it there is
selection on the dependent variable. It is carried as its own segment
("manual") and everything downstream decides explicitly whether to use it.

READ-ONLY: every database read goes through common.read_only_db (mode=ro).
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from engines.forced_trade.common import (
    DEFAULT_DB, MS_PER_DAY, MS_PER_SEC, REPO, ensure_dir, ms_to_str, read_only_db,
)

logger = logging.getLogger("forced_trade.labels")

C66_CACHE_DIR = REPO / "data" / "external" / "forced_trade" / "c66"
C66_OUT_DIR = REPO / "outputs" / "forced_trade" / "c66"
DEFAULT_COLLECTOR_LOG = REPO / "logs" / "bybit_liquidation_collector.log"

# Python logging asctime and cmd.exe %date% %time% are LOCAL wall-clock time.
# The `window :` and `GAP OPENED` lines are UTC ISO. Never mix the two without
# converting through this zone (DST-aware, not a fixed offset).
LOG_TZ = "America/Toronto"

VENUE = "bybit"
COLLECTOR = "liquidations"
SYMBOL_ASSET = {"BTCUSDT": "BTC", "ETHUSDT": "ETH", "SOLUSDT": "SOL",
                "XRPUSDT": "XRP", "ADAUSDT": "ADA", "AVAXUSDT": "AVAX"}

# Canonical `side` is the Binance ORDER side (engines/liquidation_common).
# SELL => a LONG was liquidated (forced market sell, pushes price DOWN).
# BUY  => a SHORT was liquidated (forced market buy, pushes price UP).
# DIRECTION is the sign of the forced flow: -1 forced selling, +1 forced buying.
SIDE_DIRECTION = {"SELL": -1, "BUY": 1}


# ============================================================ input snapshot ==
#
# Everything the analysis needs from crypto_data.db is pulled ONCE into
# parquet under C66_CACHE_DIR, with a manifest. The analysis then runs from the
# snapshot, so (a) re-runs are reproducible against a fixed input and (b) no
# step touches the 40 GB database during the 03:00 mirror / 03:30 backup.
# The trades bucket cache is built separately by cascade.build_bucket_cache
# (unchanged Cycle 61 code) into the same directory.

_LIQ_SQL = """
SELECT venue, symbol, timestamp, side, side_raw, price, quantity, quote_qty,
       price_basis, source, event_time, ingested_at
FROM liquidations
WHERE venue = ?
ORDER BY timestamp
"""

_GAP_SQL = """
SELECT collector, venue, timestamp, gap_end, gap_seconds, reason, detected_at
FROM collector_gaps
WHERE collector = ? AND venue = ?
ORDER BY timestamp
"""

_BOOK_SQL = """
SELECT timestamp, mid_price, best_bid, best_ask, spread_bps,
       bid_volume_top10, ask_volume_top10, order_imbalance_top10
FROM order_book_snapshots
WHERE asset = ? AND timestamp >= ?
ORDER BY timestamp
"""


def input_paths(cache_dir=C66_CACHE_DIR, assets=("BTC", "ETH")) -> dict:
    c = Path(cache_dir)
    out = {"liquidations": c / "liquidations_bybit.parquet",
           "gaps": c / "collector_gaps_liquidations_bybit.parquet",
           "log": c / "bybit_liquidation_collector.log.snapshot",
           "manifest": c / "inputs_manifest.json"}
    for a in assets:
        out[f"book_{a}"] = c / f"book_{a}.parquet"
    return out


def extract_inputs(db_path=DEFAULT_DB, cache_dir=C66_CACHE_DIR, assets=("BTC", "ETH"),
                   log_path=DEFAULT_COLLECTOR_LOG, book_lead_days: int = 2,
                   force: bool = False) -> dict:
    """Snapshot every non-trades input into parquet. Read-only against the DB.

    The book is pulled from `book_lead_days` before the first liquidation row
    (floored to UTC midnight): the T2/T3 depth statistics need a 24h trailing
    baseline in front of the first labelled event.
    """
    paths = input_paths(cache_dir, assets)
    if paths["manifest"].exists() and not force:
        logger.info("input snapshot exists (%s); not re-extracting", paths["manifest"])
        return json.loads(paths["manifest"].read_text(encoding="utf-8"))

    ensure_dir(Path(cache_dir))
    manifest = {"extracted_at_ms": int(time.time() * 1000), "db_path": str(db_path),
                "queries": {"liquidations": _LIQ_SQL.strip(), "gaps": _GAP_SQL.strip(),
                            "book": _BOOK_SQL.strip()},
                "tables": {}}
    t0 = time.time()
    with read_only_db(db_path) as conn:
        liq = pd.read_sql_query(_LIQ_SQL, conn, params=(VENUE,))
    liq.to_parquet(paths["liquidations"], index=False)
    manifest["tables"]["liquidations"] = {
        "rows": int(len(liq)), "min_ts": int(liq["timestamp"].min()),
        "max_ts": int(liq["timestamp"].max())}
    logger.info("liquidations: %d rows %s -> %s (%.1fs)", len(liq),
                ms_to_str(liq["timestamp"].min()), ms_to_str(liq["timestamp"].max()),
                time.time() - t0)

    with read_only_db(db_path) as conn:
        gaps = pd.read_sql_query(_GAP_SQL, conn, params=(COLLECTOR, VENUE))
    gaps.to_parquet(paths["gaps"], index=False)
    manifest["tables"]["collector_gaps"] = {"rows": int(len(gaps)),
                                            "open": int(gaps["gap_end"].isna().sum())}
    logger.info("collector_gaps: %d rows (%d open)", len(gaps), gaps["gap_end"].isna().sum())

    first_day = (int(liq["timestamp"].min()) // MS_PER_DAY) * MS_PER_DAY
    book_from = first_day - book_lead_days * MS_PER_DAY
    manifest["book_from_ms"] = int(book_from)
    for a in assets:
        t1 = time.time()
        with read_only_db(db_path) as conn:
            book = pd.read_sql_query(_BOOK_SQL, conn, params=(a, int(book_from)))
        book.to_parquet(paths[f"book_{a}"], index=False)
        manifest["tables"][f"book_{a}"] = {
            "rows": int(len(book)), "min_ts": int(book["timestamp"].min()),
            "max_ts": int(book["timestamp"].max())}
        logger.info("[%s] order_book_snapshots: %d rows from %s (%.1fs)", a, len(book),
                    ms_to_str(book_from), time.time() - t1)

    with read_only_db(db_path) as conn:
        spans = {}
        for t in ("trades", "order_book_snapshots"):
            for a in assets:
                mn = conn.execute(f"SELECT MIN(timestamp) FROM {t} WHERE asset=?",
                                  (a,)).fetchone()[0]
                mx = conn.execute(f"SELECT MAX(timestamp) FROM {t} WHERE asset=?",
                                  (a,)).fetchone()[0]
                spans[f"{t}:{a}"] = [mn, mx]
    manifest["spans"] = spans

    shutil.copyfile(log_path, paths["log"])
    manifest["log"] = {"source": str(log_path), "bytes": paths["log"].stat().st_size}
    manifest["extract_seconds"] = round(time.time() - t0, 1)
    paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("input snapshot written to %s (%.1fs)", cache_dir, time.time() - t0)
    return manifest


def load_inputs(cache_dir=C66_CACHE_DIR, assets=("BTC", "ETH")) -> dict:
    """Read the snapshot. Raises if extract_inputs has not been run."""
    paths = input_paths(cache_dir, assets)
    if not paths["manifest"].exists():
        raise FileNotFoundError(f"no input snapshot at {cache_dir}; run extract first")
    liq = pd.read_parquet(paths["liquidations"])
    liq["asset"] = liq["symbol"].map(SYMBOL_ASSET)
    liq["direction"] = liq["side"].str.upper().map(SIDE_DIRECTION)
    out = {"manifest": json.loads(paths["manifest"].read_text(encoding="utf-8")),
           "liquidations": liq,
           "gaps": pd.read_parquet(paths["gaps"]),
           "log_path": paths["log"]}
    for a in assets:
        b = pd.read_parquet(paths[f"book_{a}"])
        b["depth"] = b["bid_volume_top10"] + b["ask_volume_top10"]
        out[f"book_{a}"] = b
    return out


# ================================================================ intervals ==
#
# Observed time is a sorted list of disjoint half-open intervals [lo, hi) in
# epoch ms, held as two int64 arrays. Everything below is set algebra on them.

def merge_intervals(lo, hi):
    """Union of [lo, hi) intervals -> sorted, disjoint (lo, hi) arrays."""
    lo = np.asarray(lo, dtype="int64")
    hi = np.asarray(hi, dtype="int64")
    keep = hi > lo
    lo, hi = lo[keep], hi[keep]
    if lo.size == 0:
        return lo, hi
    order = np.argsort(lo, kind="mergesort")
    lo, hi = lo[order], hi[order]
    out_lo, out_hi = [int(lo[0])], [int(hi[0])]
    for a, b in zip(lo[1:], hi[1:]):
        if a <= out_hi[-1]:
            out_hi[-1] = max(out_hi[-1], int(b))
        else:
            out_lo.append(int(a))
            out_hi.append(int(b))
    return np.array(out_lo, dtype="int64"), np.array(out_hi, dtype="int64")


def subtract_intervals(lo, hi, cut_lo, cut_hi):
    """[lo, hi) minus the union of [cut_lo, cut_hi). `lo/hi` must be merged."""
    cut_lo, cut_hi = merge_intervals(cut_lo, cut_hi)
    out_lo, out_hi = [], []
    j = 0
    for a, b in zip(lo, hi):
        a, b = int(a), int(b)
        while j < len(cut_lo) and cut_hi[j] <= a:
            j += 1
        cur, k = a, j
        while k < len(cut_lo) and cut_lo[k] < b:
            if cut_lo[k] > cur:
                out_lo.append(cur)
                out_hi.append(int(min(cut_lo[k], b)))
            cur = max(cur, int(cut_hi[k]))
            if cur >= b:
                break
            k += 1
        if cur < b:
            out_lo.append(cur)
            out_hi.append(b)
    return np.array(out_lo, dtype="int64"), np.array(out_hi, dtype="int64")


def intersect_intervals(lo, hi, lo2, hi2):
    """Intersection of two merged interval sets."""
    out_lo, out_hi = [], []
    i = j = 0
    while i < len(lo) and j < len(lo2):
        a, b = max(lo[i], lo2[j]), min(hi[i], hi2[j])
        if a < b:
            out_lo.append(int(a))
            out_hi.append(int(b))
        if hi[i] < hi2[j]:
            i += 1
        else:
            j += 1
    return np.array(out_lo, dtype="int64"), np.array(out_hi, dtype="int64")


def total_ms(lo, hi) -> int:
    return int((np.asarray(hi) - np.asarray(lo)).sum())


def points_inside(lo, hi, t) -> np.ndarray:
    """Vectorised: does point t fall inside [lo, hi) for some interval?"""
    t = np.asarray(t, dtype="int64")
    if len(lo) == 0:
        return np.zeros(t.shape, dtype=bool)
    idx = np.searchsorted(lo, t, side="right") - 1
    ok = idx >= 0
    idx = np.clip(idx, 0, len(lo) - 1)
    return ok & (t < hi[idx])


def spans_inside(lo, hi, a, b) -> np.ndarray:
    """Vectorised: is the whole span [a, b] inside ONE observed interval?"""
    a = np.asarray(a, dtype="int64")
    b = np.asarray(b, dtype="int64")
    if len(lo) == 0:
        return np.zeros(a.shape, dtype=bool)
    idx = np.searchsorted(lo, a, side="right") - 1
    ok = idx >= 0
    idx = np.clip(idx, 0, len(lo) - 1)
    return ok & (b <= hi[idx])


def distance_to_unobserved(lo, hi, t) -> tuple:
    """For points inside the mask: ms back to the interval start and ms forward
    to its end. Outside the mask both are 0. Used to censor clusters."""
    t = np.asarray(t, dtype="int64")
    back = np.zeros(t.shape, dtype="int64")
    fwd = np.zeros(t.shape, dtype="int64")
    if len(lo) == 0:
        return back, fwd
    idx = np.searchsorted(lo, t, side="right") - 1
    inside = (idx >= 0) & (t < hi[np.clip(idx, 0, len(lo) - 1)])
    ii = np.clip(idx, 0, len(lo) - 1)
    back[inside] = t[inside] - lo[ii][inside]
    fwd[inside] = hi[ii][inside] - t[inside]
    return back, fwd


# ============================================================ log parsing ==

_RE_BRACKET = re.compile(
    r"^\[(\d{4})-(\d{2})-(\d{2})\s+(\d{1,2}):(\d{2}):(\d{2})\.(\d{1,2})\]\s*(.*?)\s*$")
_RE_ASC = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2}) ([A-Z]+)\s+(\S+): (.*)$")
_RE_WINDOW = re.compile(r"^\s*window\s+:\s+(\S+)\s+->\s+(\S+)\s*$")
_RE_CONTROL = re.compile(r"^\s*control frames\s+:\s+(\d+)")
_RE_GAP_OPEN = re.compile(r"GAP OPENED\s+\S+ at (\S+) \(")
_RE_GAP_CLOSE = re.compile(r"GAP CLOSED\s+\S+ (\S+) -> (\S+) \(")
_RE_FAIL = re.compile(r"^\[FAIL\]\s*(.*)$")


def iso_to_ms(s: str) -> int:
    return int(datetime.fromisoformat(s).timestamp() * 1000)


class LocalClock:
    """Local wall-clock text -> UTC epoch ms, DST-aware.

    In the repeated fall-back hour a local time is ambiguous. The log is
    written sequentially, so the occurrence that keeps time monotonic is the
    right one; `ambiguous` counts how often that rule had to be applied.
    """

    def __init__(self, tz: str = LOG_TZ):
        self.tz = ZoneInfo(tz)
        self.last = None
        self.ambiguous = 0

    def to_ms(self, y, mo, d, h, mi, s, us=0) -> int:
        naive = datetime(int(y), int(mo), int(d), int(h), int(mi), int(s), int(us))
        t0 = int(naive.replace(tzinfo=self.tz, fold=0).timestamp() * 1000)
        t1 = int(naive.replace(tzinfo=self.tz, fold=1).timestamp() * 1000)
        t = t0
        if t0 != t1:
            self.ambiguous += 1
            if self.last is not None and t0 < self.last - 60_000:
                t = t1
        self.last = t if self.last is None else max(self.last, t)
        return t


@dataclass
class Run:
    """One invocation of the scheduled collector, as the log records it."""
    idx: int
    line_no: int
    start_ms: int
    exit_code: object = None
    exit_ms: object = None
    window_start_ms: object = None
    window_end_ms: object = None
    control_frames: object = None
    fails: list = field(default_factory=list)
    events: list = field(default_factory=list)
    last_life_ms: object = None
    last_flush_ms: object = None
    segments: list = field(default_factory=list)
    status: str = ""

    @property
    def first_connect_ms(self):
        for k, t, _ in self.events:
            if k == "connect":
                return t
        return None


def parse_collector_log(path, tz: str = LOG_TZ) -> tuple:
    """Split the service log into runs. Returns (runs, clock)."""
    clock = LocalClock(tz)
    runs, cur = [], None
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.rstrip("\r\n")
            m = _RE_BRACKET.match(line)
            if m:
                y, mo, d, h, mi, s, cs, rest = m.groups()
                t = clock.to_ms(y, mo, d, h, mi, s, int(cs.ljust(2, "0")) * 10_000)
                if rest.startswith("Starting Bybit liquidation stream capture"):
                    cur = Run(idx=len(runs), line_no=ln, start_ms=t)
                    runs.append(cur)
                elif rest.startswith("Capture exited with") and cur is not None:
                    cur.exit_code = int(re.search(r"(-?\d+)", rest).group(1))
                    cur.exit_ms = t
                continue
            if cur is None:
                continue
            m = _RE_ASC.match(line)
            if m:
                y, mo, d, h, mi, s, _level, _name, msg = m.groups()
                t = clock.to_ms(y, mo, d, h, mi, s)
                cur.last_life_ms = t if cur.last_life_ms is None else max(cur.last_life_ms, t)
                if msg.startswith("flushed "):
                    cur.last_flush_ms = t
                    continue
                if "connected to wss://" in msg:
                    cur.events.append(("connect", t, None))
                    continue
                g = _RE_GAP_OPEN.search(msg)
                if g:
                    cur.events.append(("gap_open", iso_to_ms(g.group(1)), None))
                    continue
                g = _RE_GAP_CLOSE.search(msg)
                if g:
                    cur.events.append(("gap_close", iso_to_ms(g.group(1)),
                                       iso_to_ms(g.group(2))))
                continue
            m = _RE_WINDOW.match(line)
            if m:
                cur.window_start_ms = iso_to_ms(m.group(1))
                cur.window_end_ms = iso_to_ms(m.group(2))
                continue
            m = _RE_CONTROL.match(line)
            if m:
                cur.control_frames = int(m.group(1))
                continue
            m = _RE_FAIL.match(line)
            if m:
                cur.fails.append(m.group(1)[:200])
    return runs, clock


def build_run_segments(run: Run, connect_pad_ms: int = 1000,
                       disconnect_pad_ms: int = 60_000,
                       min_judgeable_sec: int = 300) -> Run:
    """Observed [lo, hi) segments for one run, written onto the run.

    connect_pad_ms   -- `connected to` is logged at 1-second resolution and
                        the subscription lands after it; a connect is treated
                        as observed only from floor-second + pad. When a GAP
                        CLOSED line follows, its ms-precise end replaces the
                        rounded time (no pad needed).
    disconnect_pad_ms -- the collector disables websocket keepalive and treats
                        a 30s recv timeout as an idle tick, so a dead socket
                        is only noticed when a send or recv finally errors.
                        Observed time therefore ends this long BEFORE each
                        logged disconnect. Conservative by construction.
    """
    ev = list(run.events)
    for i, (k, _t, x) in enumerate(ev):
        if k == "gap_close" and i > 0 and ev[i - 1][0] == "connect":
            ev[i - 1] = ("connect", x, "precise")
    segs, cur = [], None
    for k, t, x in ev:
        if k == "connect" and cur is None:
            cur = t if x == "precise" else t + connect_pad_ms
        elif k == "gap_open" and cur is not None:
            end = t - disconnect_pad_ms
            if end > cur:
                segs.append((cur, end))
            cur = None
    # A completed run flushes its buffer on exit, so it is observed to its
    # window end. A run with no window line (killed, or still running at the
    # snapshot) only PERSISTED rows up to its last `flushed` line: rows are
    # written in batches of 50 and the kline control topic (a frame every ~2s)
    # keeps the 30s idle-flush from ever firing, so anything buffered after the
    # last flush was lost (killed) or is not in the snapshot (running).
    end = run.window_end_ms if run.window_end_ms is not None else run.last_flush_ms
    if cur is not None and end is not None and end > cur:
        segs.append((cur, int(end)))
    segs = [(a, b) for a, b in segs if b > a]

    connected_sec = sum(b - a for a, b in segs) / 1000.0
    if run.exit_code is None:
        run.status = "killed_or_running"
    elif any("subscribe request(s) were rejected" in f for f in run.fails):
        run.status = "subscribe_rejected"
        segs = []
    elif (run.control_frames == 0) and connected_sec >= min_judgeable_sec:
        # Socket open, market-data subscription dead: unobserved, not quiet.
        run.status = "dead_subscription"
        segs = []
    elif run.exit_code == 2:
        run.status = "quiet_hour"          # control live, zero liquidations
    elif run.exit_code == 0:
        run.status = "ok"
    else:
        run.status = f"exit_{run.exit_code}"
    run.segments = segs
    return run


def handoff_band(runs, pad_sec: float = 2.0, q_lo: float = 0.01, q_hi: float = 0.99) -> dict:
    """The periodic blind spot between hourly runs, measured from the runs.

    Scheduled runs end late in minute :59 and the next connects early in
    minute :00. The band is [p01(end second-of-hour) - pad, p99(first-connect
    second-of-hour) + pad], measured only from runs that follow the schedule
    (end in [58:00, 60:00), connect in [00:00, 02:00)). Off-schedule runs are
    handled exactly by the log mask; the band is applied to EVERY hour so the
    blind spot is removed uniformly from events and controls alike.
    """
    ends, starts = [], []
    for r in runs:
        if r.window_end_ms is not None:
            soh = (r.window_end_ms % 3_600_000) / 1000.0
            if 3480 <= soh < 3600:
                ends.append(soh)
        c = r.first_connect_ms
        if c is not None:
            soh = (c % 3_600_000) / 1000.0
            if soh < 120:
                starts.append(soh)
    if not ends or not starts:
        return {"start_soh": None, "end_soh": None}
    e = np.array(ends)
    s = np.array(starts)
    return {
        "start_soh": float(np.quantile(e, q_lo) - pad_sec),
        "end_soh": float(np.quantile(s, q_hi) + pad_sec + 1.0),  # +1: second-floored
        "n_end": int(len(e)), "n_start": int(len(s)),
        "end_soh_min": float(e.min()), "end_soh_p50": float(np.median(e)),
        "end_soh_max": float(e.max()),
        "start_soh_min": float(s.min()), "start_soh_p50": float(np.median(s)),
        "start_soh_max": float(s.max()), "pad_sec": pad_sec,
    }


def band_intervals(band: dict, first_ms: int, last_ms: int):
    """Materialise the hourly band over [first_ms, last_ms]."""
    if band.get("start_soh") is None:
        return np.array([], dtype="int64"), np.array([], dtype="int64")
    h0 = (first_ms // 3_600_000) * 3_600_000
    hours = np.arange(h0, last_ms + 3_600_000, 3_600_000, dtype="int64")
    pre = int(round((3600.0 - band["start_soh"]) * 1000))
    post = int(round(band["end_soh"] * 1000))
    return merge_intervals(hours - pre, hours + post)


@dataclass
class Coverage:
    """The observed-time mask and how it was built."""
    lo: np.ndarray                 # scheduled observed time, band applied
    hi: np.ndarray
    log_lo: np.ndarray             # scheduled observed time from the log alone
    log_hi: np.ndarray
    manual_lo: np.ndarray          # manual 62A segment (events only)
    manual_hi: np.ndarray
    runs: list
    band: dict
    first_ms: int                  # first scheduled connect
    last_ms: int                   # analysis end
    stats: dict


def build_coverage(log_path, liq: pd.DataFrame, end_ms: int,
                   connect_pad_ms: int = 1000, disconnect_pad_ms: int = 60_000,
                   band_pad_sec: float = 2.0, apply_band: bool = True,
                   tz: str = LOG_TZ, band_extra_pre_sec: float = 0.0,
                   band_override=None) -> Coverage:
    """Observed time = union of run segments, minus the hourly handoff band.

    `end_ms` truncates everything at the analysis end (the earliest input
    snapshot), so a run still in progress cannot claim time we have no rows
    for. `band_override` = (start_soh, end_soh) replaces the measured band
    outright -- sensitivity only (the brief's bounding rule, :59:00-:00:30).
    """
    runs, clock = parse_collector_log(log_path, tz=tz)
    for r in runs:
        build_run_segments(r, connect_pad_ms, disconnect_pad_ms)

    seg = [(a, b) for r in runs for (a, b) in r.segments]
    log_lo, log_hi = merge_intervals([a for a, _ in seg], [b for _, b in seg])
    log_lo, log_hi = intersect_intervals(log_lo, log_hi, np.array([0], dtype="int64"),
                                         np.array([end_ms], dtype="int64"))
    first_ms = int(log_lo[0]) if len(log_lo) else int(end_ms)

    band = handoff_band(runs, pad_sec=band_pad_sec)
    if band.get("start_soh") is not None and band_extra_pre_sec:
        # sensitivity only: widen the band backwards into minute :58
        band["start_soh"] -= float(band_extra_pre_sec)
        band["extra_pre_sec"] = float(band_extra_pre_sec)
    if band.get("start_soh") is not None and band_override is not None:
        band["measured_start_soh"] = band["start_soh"]
        band["measured_end_soh"] = band["end_soh"]
        band["start_soh"], band["end_soh"] = (float(x) for x in band_override)
        band["override"] = True
    b_lo, b_hi = band_intervals(band, first_ms, end_ms)
    if apply_band:
        lo, hi = subtract_intervals(log_lo, log_hi, b_lo, b_hi)
    else:
        lo, hi = log_lo, log_hi

    # Manual 62A rows: everything before the first scheduled connect. Their
    # true coverage is unknown (no log), so the segment is just first..last row.
    pre = liq.loc[liq["timestamp"] < first_ms, "timestamp"]
    if len(pre):
        manual_lo = np.array([int(pre.min())], dtype="int64")
        manual_hi = np.array([int(pre.max()) + 1], dtype="int64")
    else:
        manual_lo = manual_hi = np.array([], dtype="int64")

    span = end_ms - first_ms
    status_counts = pd.Series([r.status for r in runs]).value_counts().to_dict()
    stats = {
        "first_scheduled_connect": ms_to_str(first_ms),
        "analysis_end": ms_to_str(end_ms),
        "span_hours": span / 3_600_000,
        "observed_log_hours": total_ms(log_lo, log_hi) / 3_600_000,
        "observed_hours": total_ms(lo, hi) / 3_600_000,
        "observed_frac": total_ms(lo, hi) / span if span > 0 else float("nan"),
        "band_removed_beyond_log_hours":
            (total_ms(log_lo, log_hi) - total_ms(lo, hi)) / 3_600_000,
        "n_runs": len(runs), "run_status": status_counts,
        "ambiguous_local_times": clock.ambiguous,
        "connect_pad_ms": connect_pad_ms, "disconnect_pad_ms": disconnect_pad_ms,
        "manual_segment": ([ms_to_str(manual_lo[0]), ms_to_str(manual_hi[0])]
                           if len(manual_lo) else None),
        "manual_rows": int(len(pre)),
    }
    return Coverage(lo=lo, hi=hi, log_lo=log_lo, log_hi=log_hi,
                    manual_lo=manual_lo, manual_hi=manual_hi, runs=runs, band=band,
                    first_ms=first_ms, last_ms=int(end_ms), stats=stats)
