"""Tests for the Cycle 66 labelled-validation modules (engines/forced_trade/).

Synthetic inputs only: no database, no snapshot, no network. Each test pins a
behaviour the Cycle 66 measurement depends on -- the coverage mask, the retry
chain merge, clustering and censoring, and the scoring arithmetic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from engines.forced_trade import cascade_profile as CP  # noqa: E402
from engines.forced_trade import label_inventory as LI  # noqa: E402
from engines.forced_trade import label_scoring as LS  # noqa: E402
from engines.forced_trade import labels as LB  # noqa: E402
from engines.forced_trade import signature_sizing as SS  # noqa: E402

H = 3_600_000
T0 = 1_787_155_200_000          # 2026-08-19 16:00:00 UTC (12:00 EDT)


# ---------------------------------------------------------------- intervals

def test_interval_algebra():
    lo, hi = LB.merge_intervals([0, 5, 20], [10, 12, 30])
    assert lo.tolist() == [0, 20] and hi.tolist() == [12, 30]
    s_lo, s_hi = LB.subtract_intervals(lo, hi, [2, 25], [4, 40])
    assert list(zip(s_lo, s_hi)) == [(0, 2), (4, 12), (20, 25)]
    i_lo, i_hi = LB.intersect_intervals(lo, hi, np.array([8]), np.array([22]))
    assert list(zip(i_lo, i_hi)) == [(8, 12), (20, 22)]
    assert LB.points_inside(lo, hi, [0, 11, 12, 25]).tolist() == [True, True, False, True]
    assert LB.spans_inside(lo, hi, [1, 1], [12, 13]).tolist() == [True, False]


def test_local_clock_is_dst_aware():
    c = LB.LocalClock("America/Toronto")
    assert c.to_ms(2026, 8, 19, 12, 0, 0) == T0                      # EDT = UTC-4
    assert c.to_ms(2026, 12, 1, 12, 0, 0) == 1_796_144_400_000       # EST = UTC-5
    # fall-back hour: 01:30 occurs twice; sequential lines must stay monotonic
    c2 = LB.LocalClock("America/Toronto")
    first = c2.to_ms(2026, 11, 1, 1, 50, 0)
    second = c2.to_ms(2026, 11, 1, 1, 10, 0)     # after the clock went back
    assert second > first and c2.ambiguous == 2


# --------------------------------------------------------------- log parse

LOG = """\
[2026-08-19 12:00:00.47] Starting Bybit liquidation stream capture...
2026-08-19 12:00:01 INFO    collector.liquidations.bybit: connected to wss://stream.bybit.com/v5/public/linear
2026-08-19 12:10:00 INFO    collector.liquidations.bybit: flushed 50 events (written=50 dupes=0, run total=50)
2026-08-19 12:20:00 WARNING collector: GAP OPENED  liquidations/bybit at 2026-08-19T16:20:00.000000+00:00 (gaierror: x)
2026-08-19 12:20:04 WARNING collector: GAP OPENED  liquidations/bybit at 2026-08-19T16:20:04.000000+00:00 (gaierror: x)
2026-08-19 12:21:05 INFO    collector.liquidations.bybit: connected to wss://stream.bybit.com/v5/public/linear
2026-08-19 12:21:05 WARNING collector: GAP CLOSED  liquidations/bybit 2026-08-19T16:20:04.000000+00:00 -> 2026-08-19T16:21:05.500000+00:00 (61.5s, gaierror: x)
  window             : 2026-08-19T16:00:00.603000+00:00 -> 2026-08-19T16:59:10.000000+00:00
  control frames     : 1500 (kline.1.BTCUSDT)
[2026-08-19 12:59:11.00] Capture exited with 0.
[2026-08-19 13:00:01.00] Starting Bybit liquidation stream capture...
2026-08-19 13:00:02 INFO    collector.liquidations.bybit: connected to wss://stream.bybit.com/v5/public/linear
2026-08-19 13:05:00 INFO    collector.liquidations.bybit: flushed 50 events (written=50 dupes=0, run total=50)
2026-08-19 13:06:00 INFO    collector.liquidations.bybit: subscribe -> something
[2026-08-19 15:00:01.00] Starting Bybit liquidation stream capture...
2026-08-19 15:00:02 INFO    collector.liquidations.bybit: connected to wss://stream.bybit.com/v5/public/linear
  window             : 2026-08-19T19:00:00.500000+00:00 -> 2026-08-19T19:59:10.000000+00:00
  control frames     : 0 (kline.1.BTCUSDT)
[2026-08-19 15:59:11.00] Capture exited with 2.
"""


@pytest.fixture
def runs(tmp_path):
    p = tmp_path / "c.log"
    p.write_text(LOG, encoding="utf-8")
    rs, _ = LB.parse_collector_log(p)
    for r in rs:
        LB.build_run_segments(r, connect_pad_ms=1000, disconnect_pad_ms=60_000)
    return rs


def test_log_runs_statuses(runs):
    assert [r.status for r in runs] == ["ok", "killed_or_running", "dead_subscription"]
    assert runs[0].window_end_ms == T0 + 3550_000


def test_segments_respect_pads_and_precise_reconnect(runs):
    segs = runs[0].segments
    # first connect 16:00:01 local-floor + 1s pad; observation ends 60s BEFORE
    # the first GAP OPENED of the retry chain; resumes at the precise GAP
    # CLOSED instant; ends at the window end.
    assert segs[0] == (T0 + 2000, T0 + 20 * 60_000 - 60_000)
    assert segs[1] == (T0 + 21 * 60_000 + 5500, T0 + 3550_000)


def test_killed_run_ends_at_last_flush(runs):
    # rows after the last flush were never persisted, whatever the log says
    assert runs[1].segments == [(T0 + H + 3000, T0 + H + 5 * 60_000)]


def test_dead_subscription_is_unobserved(runs):
    assert runs[2].segments == []


def test_retry_chain_merge():
    gaps = pd.DataFrame({
        "timestamp": [100_000, 104_000, 112_000, 500_000],
        "gap_end": [None, None, 170_000, 502_000],
        "reason": ["dns", "dns", "dns", "closed"]})
    ch = LI.merge_gap_chains(gaps)
    assert len(ch) == 2
    assert ch.iloc[0]["start_ms"] == 100_000 and ch.iloc[0]["end_ms"] == 170_000
    assert ch.iloc[0]["open_rows"] == 2


def test_handoff_band_is_measured_from_runs(runs):
    band = LB.handoff_band(runs, pad_sec=2.0)
    assert 3540 < band["start_soh"] < 3560        # :59:10 minus pad
    assert 0 < band["end_soh"] < 10               # :00:01-:00:02 plus pad


def test_band_override_replaces_measured_band(tmp_path):
    p = tmp_path / "c.log"
    p.write_text(LOG, encoding="utf-8")
    liq = pd.DataFrame({"timestamp": [T0 + 600_000]})
    cov = LB.build_coverage(p, liq, T0 + 4 * H, band_override=(3540, 30))
    assert (cov.band["start_soh"], cov.band["end_soh"]) == (3540.0, 30.0)
    assert cov.band["override"] and 3540 < cov.band["measured_start_soh"] < 3560
    # :59:00-:00:30 around 17:00 UTC is gone from the observed mask
    assert not LB.points_inside(cov.lo, cov.hi, [T0 + H - 59_000, T0 + H + 29_000]).any()
    assert LB.points_inside(cov.lo, cov.hi, [T0 + H + 31_000]).all()


# --------------------------------------------------- handoff residual tests

def _hourly_cov(hours, gap_lo_sec=-50, gap_hi_sec=5, band=(3546.0, 10.0)):
    """Observed everywhere except [H + gap_lo, H + gap_hi) around each hour."""
    hs = T0 + np.arange(1, hours, dtype="int64") * H
    lo = np.r_[T0 + gap_hi_sec * 1000, hs + gap_hi_sec * 1000]
    hi = np.r_[hs + gap_lo_sec * 1000, T0 + hours * H + gap_lo_sec * 1000]
    cov = _cov(lo, hi, T0)
    cov.band = {"start_soh": band[0], "end_soh": band[1]}
    return cov


def test_local_baseline_is_one_when_mask_is_right_and_fails_when_it_is_not():
    cov = _hourly_cov(48)
    secs = np.concatenate([np.arange(a, b, 1000) for a, b in zip(cov.lo, cov.hi)])
    ok, _ = LI.handoff_local_baseline(pd.DataFrame({"timestamp": secs}), cov,
                                      adj_secs=(60,), n_boot=20)
    assert np.allclose(ok["ratio"], 1.0) and np.allclose(ok["ratio_head"], 1.0)
    # the incoming run is really dead 5s longer than the log says: the head of
    # the band loses all its rows and the ratio must drop to tail/(tail+head)
    soh = (secs // 1000) % 3600
    dead = pd.DataFrame({"timestamp": secs[~((soh >= 5) & (soh < 10))]})
    bad, _ = LI.handoff_local_baseline(dead, cov, adj_secs=(60,), n_boot=20)
    r = bad[(bad["unit"] == "rows") & (bad["baseline"] == "local +-60s")].iloc[0]
    assert r["ratio_head"] == 0.0 and r["ratio_tail"] == pytest.approx(1.0)
    assert r["ratio"] == pytest.approx(4 / 9)     # tail 3546-3550, head 5-10


def test_leading_edge_floor_is_measured_from_runs():
    runs = []
    for k in range(6):
        h = T0 + k * H
        r = LB.Run(idx=k, line_no=k, start_ms=h + 500, window_end_ms=h + 3_550_000,
                   events=[("connect", h + 2_000, None)], status="ok")
        runs.append(r)
    cov = _hourly_cov(6)
    cov.runs = runs
    rows = pd.DataFrame({"timestamp": [r.first_connect_ms + 1_500 for r in runs]})
    df, ramp, s = LI.leading_edge_latency(rows, cov)
    assert s["n_handoff_runs"] == 5               # the first run has no predecessor
    assert s["floor_first_row_soh"] == pytest.approx(3.5)
    assert s["floor_past_band_end_sec"] == pytest.approx(3.5 - 10.0)
    assert np.allclose(df["first_row_minus_connect_sec"], 1.5)


def test_day_cluster_se():
    rng = np.random.default_rng(3)
    v = rng.normal(size=12)
    # one event per day: the clustered SE is exactly the iid SE
    assert CP.day_cluster_se(v, np.arange(12)) == pytest.approx(v.std(ddof=1) / np.sqrt(12))
    # duplicating every event inside its own day adds no information: the
    # clustered SE must not shrink (the iid SE would, by sqrt 2)
    days = np.arange(12)
    assert CP.day_cluster_se(np.r_[v, v], np.r_[days, days]) == pytest.approx(
        CP.day_cluster_se(v, days))


# ---------------------------------------------------------------- clusters

def _cov(lo, hi, first_ms):
    lo = np.asarray(lo, dtype="int64")
    hi = np.asarray(hi, dtype="int64")
    return LB.Coverage(lo=lo, hi=hi, log_lo=lo, log_hi=hi,
                       manual_lo=np.array([], "int64"), manual_hi=np.array([], "int64"),
                       runs=[], band={}, first_ms=first_ms, last_ms=int(hi[-1]), stats={})


def test_clusters_gap_censoring_and_manual():
    base = 1_000_000_000
    cov = _cov([base], [base + 3_600_000], base)
    ts = [base - 500_000,                       # before first scheduled connect -> manual
          base + 5_000,                         # 5s after coverage start -> censored left
          base + 100_000, base + 110_000, base + 125_000,   # one cluster at gap 30s
          base + 3_590_000]                     # 10s before coverage end -> censored right
    liq = pd.DataFrame({"symbol": "BTCUSDT", "timestamp": ts, "quantity": 1.0,
                        "quote_qty": 100.0, "direction": [-1, -1, 1, 1, -1, 1]})
    grids = {"BTC": {"first": base - 600_000, "step": 10_000, "n": 500,
                     "px": np.full(500, 50_000.0)}}
    c = LI.cluster_labels(liq, 30, cov, grids)
    assert len(c) == 4
    mid = c[c["n_events"] == 3].iloc[0]
    assert not mid["censored"] and mid["direction"] == 1
    assert mid["notional_spot"] == pytest.approx(150_000.0)
    assert c.iloc[0]["segment"] == "manual" and c.iloc[0]["censored"]
    assert c[c["start_ms"] == base + 5_000].iloc[0]["censored_left"]
    assert c[c["start_ms"] == base + 3_590_000].iloc[0]["censored_right"]


# ----------------------------------------------------------------- scoring

def test_window_confusion_arithmetic():
    n = 1600
    ts = np.arange(n, dtype="int64") * 60_000
    d = pd.DataFrame({"ts_ms": ts, "med_dom": 1.0, "sd_ret": 1.0,
                      "cand": np.zeros(n, dtype=bool)})
    d.loc[[1500, 1510], "cand"] = True          # one hit, one false alarm
    cl = pd.DataFrame({"symbol": "BTCUSDT", "segment": "scheduled", "censored": False,
                       "size_usd": [2e6], "start_ms": [1500 * 60_000 + 5_000],
                       "end_ms": [1500 * 60_000 + 20_000]})
    r = LS.window_confusion(d, 60, cl, 1e6, 0, np.array([0]), np.array([n * 60_000]),
                            n_boot=0)
    assert (r["TP"], r["FP"], r["FN"]) == (1, 1, 0)
    assert r["precision"] == pytest.approx(0.5)
    assert r["prevalence"] == pytest.approx(1 / n)
    assert r["lift"] == pytest.approx(0.5 * n)


def test_strict_vs_overlap_recall():
    d = pd.DataFrame({"ts_ms": np.array([0, 60_000, 120_000, 180_000], dtype="int64"),
                      "cand": [False, False, True, False]})
    ev = pd.DataFrame({"start_ms": [120_000], "end_ms": [180_000]})
    # a 3-minute cluster starting at t=0: detector fires inside it (overlap)
    # but not within 60s of its onset (strict)
    cl = pd.DataFrame({"start_ms": [0], "end_ms": [170_000]})
    h = LS.label_to_detector(cl, d, ev, 60, 60)
    assert bool(h["hit_overlap"].iloc[0]) and not bool(h["hit_strict"].iloc[0])


def test_matched_controls_share_hour_and_vol_bucket():
    grid = {"first": 0, "step": 10_000, "n": 10 * 8640, "px": np.ones(10 * 8640)}
    rng = np.random.default_rng(1)
    vol = rng.random(grid["n"])
    pool = np.arange(0, grid["n"] * 10_000 - 60_000, 60_000, dtype="int64")
    anchors = np.array([5 * 3_600_000 + 120_000], dtype="int64")
    mc = LS.matched_controls(anchors, pool, grid, vol, k=10, n_vol_buckets=5, seed=2)
    assert len(mc) == 10
    assert set(((mc["anchor_ms"] // 3_600_000) % 24).tolist()) == {5}
    assert mc["vol_bucket"].nunique() == 1


# ---------------------------------------------------------------- T4 sizes

def test_signature_term_counts():
    assert SS.sig_terms(5, 4) == 780           # the brief's number
    assert SS.logsig_terms(5, 4) == 205        # Witt / Lyndon words
    assert SS.logsig_terms(2, 4) == 8
    assert SS.sig_terms(10, 3) == 1110
