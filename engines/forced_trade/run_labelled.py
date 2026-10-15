#!/usr/bin/env python
"""
engines/forced_trade/run_labelled.py

Cycle 66 -- A2 validation against labelled (Bybit) liquidations. CLI.

    python -m engines.forced_trade.run_labelled extract       # DB -> snapshot (read-only)
    python -m engines.forced_trade.run_labelled t1-inventory
    python -m engines.forced_trade.run_labelled t2-confusion
    python -m engines.forced_trade.run_labelled t3-profile
    python -m engines.forced_trade.run_labelled t4-sizing --tooling-results <json>
    python -m engines.forced_trade.run_labelled contiguity    # DB, targeted reads
    python -m engines.forced_trade.run_labelled all

MEASUREMENT ONLY: no strategy, no backtest, no Sharpe, no detector retuning.
Only `extract` and `contiguity` read crypto_data.db (mode=ro); everything else
runs from the snapshot in data/external/forced_trade/c66/. Both DB commands
refuse to run inside the mirror/backup guard window (default 02:50-03:45
local) unless --ignore-time-guard is given.

Rule 25: --validate and -vv are the defaults.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from engines.forced_trade import cascade as C
from engines.forced_trade import cascade_profile as CP
from engines.forced_trade import label_inventory as LI
from engines.forced_trade import label_scoring as LS
from engines.forced_trade import labels as LB
from engines.forced_trade import signature_sizing as SS
from engines.forced_trade.common import (
    DEFAULT_CACHE_DIR, DEFAULT_DB, banner, ensure_dir, fmt_table, ms_to_str,
    read_only_db, setup_logging,
)

logger = logging.getLogger("forced_trade.run_labelled")

ASSET_SYMBOL = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}


# ------------------------------------------------------------------ utils --

def _save(obj, out_dir: Path, name: str) -> Path:
    ensure_dir(out_dir)
    if isinstance(obj, pd.DataFrame):
        p = out_dir / f"{name}.csv"
        obj.to_csv(p, index=False)
    else:
        p = out_dir / f"{name}.json"
        p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", p)
    return p


def _save_rows(df: pd.DataFrame, args, name: str) -> Path:
    """Row-level tables (one row per cluster) are tens of MB: they go to the
    gitignored cache as parquet; outputs/ keeps the summaries."""
    p = ensure_dir(Path(args.rows_dir or (Path(args.cache_dir) / "rows"))) / f"{name}.parquet"
    df.to_parquet(p, index=False)
    logger.info("wrote %s", p)
    return p


def _f(x, nd=2):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "n/a"
        return f"{x:.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _pct(x, nd=1):
    try:
        return "n/a" if x is None or np.isnan(x) else f"{100 * x:.{nd}f}%"
    except TypeError:
        return "n/a"


def _usd(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(x) >= div:
            return f"${x / div:.1f}{suf}"
    return f"${x:.0f}"


def time_guard(args) -> None:
    """Refuse DB reads inside the mirror/backup window (local wall clock)."""
    if args.ignore_time_guard:
        return
    lo, hi = args.guard_window.split("-")
    now = datetime.now().strftime("%H:%M")
    if lo <= now <= hi:
        print(f"\n[REFUSED] local time {now} is inside the DB guard window {lo}-{hi} "
              f"(03:00 mirror / 03:30 backup). Re-run after {hi}, or pass "
              f"--ignore-time-guard.", file=sys.stderr)
        raise SystemExit(3)


# ---------------------------------------------------------------- context --

class Ctx:
    """Everything derived from the snapshot, built once per invocation."""

    def __init__(self, args):
        t0 = time.time()
        self.args = args
        self.out = Path(args.out_dir)
        self.inp = LB.load_inputs(args.cache_dir, tuple(args.assets))
        self.liq_all = self.inp["liquidations"]
        self.buckets = {}
        for a in args.assets:
            p = C.bucket_cache_path(a, args.bucket_sec, args.cache_dir)
            if not p.exists():
                raise FileNotFoundError(f"bucket cache missing: {p} (run extract)")
            self.buckets[a] = pd.read_parquet(p)
        end_ms = min(int(b["bucket_start_ms"].max()) + args.bucket_sec * 1000
                     for b in self.buckets.values())
        end_ms = min(end_ms, int(self.inp["manifest"]["extracted_at_ms"]))
        self.end_ms = end_ms
        self.liq = self.liq_all[self.liq_all["timestamp"] < end_ms].copy()
        self.cov = LB.build_coverage(self.inp["log_path"], self.liq, end_ms,
                                     connect_pad_ms=args.connect_pad_ms,
                                     disconnect_pad_ms=args.disconnect_pad_ms,
                                     band_pad_sec=args.band_pad_sec,
                                     apply_band=args.apply_band, tz=args.log_tz,
                                     band_extra_pre_sec=args.band_extra_pre_sec,
                                     band_override=args.band_soh)
        self.sched = self.liq[self.liq["timestamp"] >= self.cov.first_ms]
        self.grids = {a: LI.spot_grid(b, args.bucket_sec) for a, b in self.buckets.items()}
        self.books = {a: self.inp[f"book_{a}"] for a in args.assets}
        self.clusters = {}
        for g in args.cluster_gaps:
            self.clusters[g] = LS.add_buckets(
                LI.cluster_labels(self.liq, g, self.cov, self.grids))
        self._det = {}
        logger.info("context ready in %.1fs (analysis end %s)", time.time() - t0,
                    ms_to_str(end_ms))

    def asset_clusters(self, asset: str, gap: float) -> pd.DataFrame:
        c = self.clusters[gap]
        return c[c["symbol"] == ASSET_SYMBOL[asset]]

    def detection(self, asset: str, name: str, p: C.CascadeParams):
        key = (asset, name, p.window_sec)
        if key not in self._det:
            self._det[key] = LS.detect_full(self.buckets[asset], p, self.args.bucket_sec)
        return self._det[key]

    def exclusion_spans(self):
        """Controls stay this far from any cluster >= exclusion floor (all
        symbols, all segments) -- a control must be cascade-free."""
        c = self.clusters[self.args.primary_gap]
        big = c[c["size_usd"] >= self.args.exclusion_floor_usd]
        pad = self.args.exclusion_sec * 1000
        return LB.merge_intervals(big["start_ms"].values - pad, big["end_ms"].values + pad)


# ================================================================== T1 =====

def cmd_t1(args, ctx: Ctx) -> int:
    out = ctx.out
    cov = ctx.cov
    print(banner("T1 -- INVENTORY OF THE LABELLED DATA"))

    print("\nVENUE INVENTORY (one line each):")
    m = ctx.inp["manifest"]
    lines = [
        "trades               : Binance SPOT BTC/USDT, ETH/USDT -- ccxt REST fetch_trades, "
        "fromId cursor (engines/crypto_data_collector.py:1156 TRADES_SYMBOL_MAP); "
        "no venue column; BTC+ETH only",
        "order_book_snapshots : Binance SPOT top-10 REST snapshots every 10s "
        "(crypto_data_collector.py:1003 fetch_order_book limit=10; ts = local clock "
        "when the venue sends none); BTC+ETH only",
        "liquidations         : Bybit USDT-linear PERP allLiquidation stream, "
        "BTC ETH SOL XRP ADA AVAX (stream:bybit)",
        "Binance PERP trades  : NO TABLE EXISTS in crypto_data.db or any data/*.db "
        "-- no third detector scan; the spot-vs-perp venue confound stands",
    ]
    for ln in lines:
        print("  " + ln)
    _save({"lines": lines, "spans": m.get("spans")}, out, "t1_venue_inventory")

    liq = ctx.liq
    rows, lr = [], []
    for sym, g in liq.groupby("symbol"):
        man = g[g["timestamp"] < cov.first_ms]
        asset = LB.SYMBOL_ASSET.get(sym)
        spot = (float((g["quantity"].values * LI.spot_at(ctx.grids[asset],
                                                          g["timestamp"].values)).sum())
                if asset in ctx.grids else np.nan)
        lr.append({"symbol": sym, "rows": int(len(g)), "first_utc": ms_to_str(g["timestamp"].min()),
                   "last_utc": ms_to_str(g["timestamp"].max()), "manual_rows": int(len(man)),
                   "forced_sell_frac": float((g["direction"] < 0).mean()),
                   "quote_raw_usd": float(g["quote_qty"].sum()), "qty_x_spot_usd": spot})
        rows.append([sym, f"{len(g):,}", ms_to_str(g["timestamp"].min()),
                     ms_to_str(g["timestamp"].max()), len(man),
                     f"{(g['direction'] < 0).mean():.1%}", _usd(g["quote_qty"].sum()), _usd(spot)])
    print(f"\nLABEL ROWS (venue=bybit, analysis end {ms_to_str(ctx.end_ms)}):")
    print(fmt_table(rows, ["symbol", "rows", "first", "last", "manual_rows",
                           "forced_sell%", "quote_raw", "qty x spot"]))
    print(f"  total {len(liq):,} rows; Bybit has no all-market topic, so anything "
          f"outside these 6 symbols is UNOBSERVED, not quiet.\n  quote_raw = Bybit "
          f"bankruptcy price x qty (biased, see T1.4); qty x spot = Binance spot last trade "
          f"(BTC/ETH only).")
    _save(pd.DataFrame(lr), out, "t1_label_rows")

    s = cov.stats
    runs = LI.run_table(cov)
    holes = LI.handoff_holes(cov)
    miss, outages, n_slots = LI.missing_slots(cov)
    chains = LI.merge_gap_chains(ctx.inp["gaps"])
    lout = LI.log_outages(cov)
    xc = LI.gap_crosscheck(chains, lout)
    print(banner("T1.1  COVERAGE MASK (built from the collector LOG, not collector_gaps)"))
    print(f"  span (first scheduled connect -> end) : {s['span_hours']:.1f} h  "
          f"[{s['first_scheduled_connect']} -> {s['analysis_end']}]")
    print(f"  observed per log                      : {s['observed_log_hours']:.1f} h")
    print(f"  observed after hourly band            : {s['observed_hours']:.1f} h "
          f"({_pct(s['observed_frac'])})  -- band removed a further "
          f"{s['band_removed_beyond_log_hours']:.2f} h")
    print(f"  runs                                  : {s['n_runs']}  {s['run_status']}")
    print(f"  hourly slots / with no run start      : {n_slots} / {len(miss)} "
          f"(in {len(outages)} outages)")
    print(f"  handoff holes between runs            : n={holes.get('n')} "
          f"total {holes.get('total_hours', 0):.2f} h, p50 {_f(holes.get('p50_sec'), 1)}s, "
          f"p95 {_f(holes.get('p95_sec'), 1)}s, max {_f(holes.get('max_sec'), 1)}s")
    print(f"  in-run outages (collector_gaps)       : {len(ctx.inp['gaps'])} rows -> "
          f"{len(chains)} outages after merging retry chains "
          f"({xc.get('open_rows_absorbed')} open rows absorbed, "
          f"{xc.get('unresolved_chains')} unresolved)")
    print(f"  cross-check vs log outages            : {xc.get('matched')}/{xc.get('chains')} "
          f"matched, max start diff {xc.get('max_start_diff_ms')} ms, "
          f"max end diff {xc.get('max_end_diff_ms')} ms")
    print(f"  pads: connect +{args.connect_pad_ms} ms, disconnect -{args.disconnect_pad_ms} ms "
          f"(dead-socket detection latency); manual 62A segment "
          f"{s['manual_segment']} ({s['manual_rows']} rows, EVENT-ONLY)")
    if len(outages):
        print("\n  Missing-run outages (hourly slots with no run start):")
        print(fmt_table([[o.first_slot_utc, o.last_slot_utc, o.hours, o.prev_run_status,
                          o.prev_run_log_line, o.next_run_start_utc]
                         for o in outages.itertuples()],
                        ["first_slot", "last_slot", "hours", "prev_run", "prev_log_line",
                         "next_start"]))
    _save(runs, out, "t1_runs")
    _save(outages, out, "t1_missing_run_outages")
    _save(chains, out, "t1_gap_chains")
    _save({"coverage": s, "band": cov.band, "handoff_holes": holes, "missing_slots": len(miss),
           "hourly_slots": n_slots, "gap_crosscheck": xc}, out, "t1_coverage")

    print(banner("T1.2  MASK VALIDATION"))
    om = LI.outside_mask_rows(ctx.sched, cov, args.disconnect_pad_ms, args.edge_tol_ms)
    # classify the residue: rows just BEFORE a (re)connect are pre-connect
    # delivery (Bybit pushes the last batch on subscribe / sub-second clocks)
    t = ctx.sched["timestamp"].values.astype("int64")
    inside = LB.points_inside(cov.log_lo, cov.log_hi, t)
    pads = [(g - args.disconnect_pad_ms, g) for r in cov.runs for k, g, _ in r.events
            if k == "gap_open"]
    p_lo, p_hi = LB.merge_intervals([a for a, _ in pads], [b for _, b in pads])
    resid = (~inside) & (~LB.points_inside(p_lo, p_hi, t))
    nxt = cov.log_lo[np.clip(np.searchsorted(cov.log_lo, t[resid], side="right"), 0,
                             len(cov.log_lo) - 1)]
    pre = (nxt - t[resid]) / 1000.0
    om["residue_sec_before_next_segment"] = {
        "n": int(resid.sum()), "max": float(pre.max()) if len(pre) else None,
        "n_within_5s": int((pre <= 5).sum())}
    print(f"  scheduled rows                       : {om['scheduled_rows']:,}")
    print(f"  outside the log mask                 : {om['outside_log_mask']:,}")
    print(f"    in the pre-disconnect pad (expected): {om['outside_in_disconnect_pad']:,}")
    print(f"    within {args.edge_tol_ms} ms of a mask edge        : "
          f"{om['outside_within_tol_of_edge']:,}")
    print(f"    otherwise                           : {om['outside_UNEXPLAINED']:,}  "
          f"-> residue lies <= {_f(om['residue_sec_before_next_segment']['max'], 1)}s "
          f"BEFORE a logged (re)connect ({om['residue_sec_before_next_segment']['n_within_5s']}"
          f" of {om['residue_sec_before_next_segment']['n']} within 5s)")
    print("  READ: every out-of-mask row sits in time the mask EXCLUDED conservatively "
          "(pad / pre-connect fringe).\n  No row lands in a stretch the mask calls "
          "unobserved for any other reason.")
    lat = LI.disconnect_latency(ctx.sched, cov, seed=args.seed)
    print(f"  pre-disconnect row lag p25/p50/p75   : "
          f"{[_f(x, 0) for x in lat.get('pre_disconnect_lag_sec_p25_p50_p75', [])]} s vs "
          f"random observed instant {[_f(x, 0) for x in lat.get('random_instant_lag_sec_p25_p50_p75', [])]} s")
    om["disconnect_latency"] = lat

    hv, hs = LI.handoff_validation(ctx.sched, cov, bin_sec=10)
    print(f"\n  HANDOFF BAND {hs['band'][0]:.1f}s -> {hs['band'][1]:.1f}s of the hour "
          f"(:{int(hs['band'][0] // 60):02d}:{hs['band'][0] % 60:04.1f} -> "
          f":00:{hs['band'][1]:04.1f}), applied to EVERY hour")
    print(f"  liquidation-row deficit inside the band: rows {hs['rows_in_band']:,} vs "
          f"{hs['rows_expected_in_band_if_collector_had_run']:,.0f} expected at the "
          f"out-of-band rate -> deficit {_pct(hs['observed_deficit_frac'])} "
          f"(log mask predicts {_pct(hs['mask_predicted_deficit_frac'])})")
    sel = hv[(hv["soh_start_sec"] >= 3500) | (hv["soh_start_sec"] < 30)]
    print(fmt_table([[f":{int(r.soh_start_sec // 60):02d}:{int(r.soh_start_sec % 60):02d}",
                      r.rows, _f(r.exposure_frac, 3), _f(r.rows_per_exposed_hour, 1),
                      "BAND" if r.in_band else ""] for r in sel.itertuples()],
                    ["bin", "rows", "observed_frac", "rows/obs_hour", ""]))
    # spot-tape control: is the pre-hour dip collector or market?
    sp = []
    for a, b in ctx.buckets.items():
        bb = b[(b["bucket_start_ms"] >= cov.first_ms) & (b["bucket_start_ms"] < ctx.end_ms)]
        soh = (bb["bucket_start_ms"] // 1000) % 3600
        g = bb.groupby(soh)["n_trades"].mean()
        base = g.mean()
        sp.append([a] + [_f(g.get(x, np.nan) / base, 2) for x in
                         (3450, 3480, 3510, 3530, 3540, 3560, 3580, 0, 10, 20)])
    print("\n  Independent control -- Binance SPOT trade count by bin / hour mean "
          "(trades are cursor-collected, no handoff loss):")
    print(fmt_table(sp, ["asset", ":57:30", ":58:00", ":58:30", ":58:50", ":59:00",
                         ":59:20", ":59:40", ":00:00", ":00:10", ":00:20"]))
    print("  READ: the market itself is quiet before the hour and spikes at :00:00 --"
          " the spike sits INSIDE the\n  blind spot, which is why the band is masked"
          " out of events and controls alike.")
    _save(hv, out, "t1_handoff_validation_bins")
    _save({"outside_mask": om, "handoff": hs}, out, "t1_mask_validation")

    # The observed deficit exceeds the mask's prediction: two tests.
    print(banner("T1.2b  HANDOFF RESIDUAL -- deficit beyond the mask's prediction"))
    le_runs, le_ramp, le = LI.leading_edge_latency(ctx.sched, cov)
    if le.get("n_handoff_runs"):
        q = le["first_row_soh_q"]
        print(f"  TEST 1  leading-edge latency: {le['n_handoff_runs']} on-schedule handoffs, "
              f"{le['n_with_first_row']} with a row inside the run")
        print(f"    connect line, second of hour p01/p50/p99 : "
              f"{' / '.join(_f(v, 2) for v in le['connect_soh_q'].values())} s")
        print(f"    start line -> connect line p50/p99       : "
              f"{_f(le['connect_minus_start_sec_q']['p50'], 2)} / "
              f"{_f(le['connect_minus_start_sec_q']['p99'], 2)} s")
        print(f"    first row, second of hour p01/p05/p10/p25/p50 : "
              f"{' / '.join(_f(q[k], 1) for k in ('p01', 'p05', 'p10', 'p25', 'p50'))} s")
        print(f"    first row - connect line p01/p05           : "
              f"{_f(le['first_row_minus_connect_sec_q']['p01'], 2)} / "
              f"{_f(le['first_row_minus_connect_sec_q']['p05'], 2)} s")
        print(f"    FLOOR (p{int(le['floor_q'] * 100):02d} first row) :00:"
              f"{le['floor_first_row_soh']:04.1f} vs band edge :00:{le['band_end_soh']:04.1f} "
              f"-> {le['floor_past_band_end_sec']:+.1f} s; {le['runs_first_row_before_band_end']}"
              f" runs logged a row before the band edge")
        print(f"    ramp: rows per run-second after the connect line vs the same "
              f"{le['ramp_runs']} runs at {le['ramp_ref_sec'][0]}-{le['ramp_ref_sec'][1]} s:")
        print(fmt_table([[f"{r.from_sec:+.0f}..{r.to_sec:+.0f}s", r.rows, _f(r.rel_to_ref, 2)]
                         for r in le_ramp.itertuples()], ["after connect", "rows", "rel"]))
        _save(le_runs, out, "t1_handoff_leading_edge_runs")
        _save(le_ramp, out, "t1_handoff_leading_edge_ramp")
    lb, lbs = LI.handoff_local_baseline(ctx.sched, cov, tuple(args.local_baseline_secs),
                                        n_boot=args.handoff_boot, seed=args.seed)
    print(f"\n  TEST 2  band [{lbs['band_seconds'][0]}s, {lbs['band_seconds'][1]}s) against the "
          f"adjacent seconds on the SAME hours; 90% CIs resample {lbs['days']} UTC days")
    print(f"    band {lbs['band_calendar_hours']:.2f} h, log-exposed "
          f"{lbs['band_log_exposed_hours']:.2f} h (outgoing tail "
          f"{lbs['tail_log_exposed_hours']:.2f} h, incoming head "
          f"{lbs['head_log_exposed_hours']:.2f} h)")
    print(fmt_table([[r.unit, r.baseline, f"{r.observed:,.0f}",
                      f"{r.expected_given_exposure:,.0f}",
                      f"{_f(r.ratio)} [{_f(r.ratio_ci90_lo)}, {_f(r.ratio_ci90_hi)}]",
                      f"{_f(r.ratio_tail)} [{_f(r.ratio_tail_ci90_lo)}, {_f(r.ratio_tail_ci90_hi)}]",
                      f"{_f(r.ratio_head)} [{_f(r.ratio_head_ci90_lo)}, {_f(r.ratio_head_ci90_hi)}]",
                      f"{_pct(r.observed_deficit)} vs {_pct(r.mask_predicted_deficit)}"]
                     for r in lb.itertuples()],
                    ["unit", "baseline", "obs", "exp|exposure", "ratio [ci90]",
                     "tail [ci90]", "head [ci90]", "deficit obs vs mask"]))
    print("  ratio = observed / expected given the log mask's exposure; 1.0 means the mask "
          "accounts for the whole deficit.\n  tail = outgoing run (yardstick: the window "
          "before the band); head = incoming run (the window after).")
    _save(lb, out, "t1_handoff_local_baseline")
    _save({"leading_edge": le, "local_baseline": lbs}, out, "t1_handoff_residual")

    print(banner("T1.3  CLUSTERS (gap-based, per symbol; primary gap "
                 f"{args.primary_gap:g}s pre-registered)"))
    allsum = []
    for g, c in ctx.clusters.items():
        sm = LI.cluster_summary(c, tuple(args.floors_usd))
        allsum.append(sm)
        print(f"\n  gap <= {g:g}s:")
        print(fmt_table([[r.symbol, r.clusters_all, r.scheduled_uncensored, r.censored,
                          r.manual, _f(r.n_p50, 0), _f(r.n_p90, 0), _f(r.n_p99, 0), r.n_max,
                          _usd(r.usd_p50), _usd(r.usd_p90), _usd(r.usd_max),
                          _f(r.dur_p90_sec, 0)]
                         + [getattr(r, f"n_ge_{int(fl):d}") for fl in args.floors_usd]
                         for r in sm.itertuples()],
                        ["symbol", "all", "sched_unc", "cens", "man", "n_p50", "n_p90",
                         "n_p99", "n_max", "usd_p50", "usd_p90", "usd_max", "dur_p90"]
                        + [f">={_usd(fl)}" for fl in args.floors_usd]))
        _save_rows(c, args, f"t1_clusters_gap{int(g)}")
    _save(pd.concat(allsum), out, "t1_cluster_summary")
    print("  size basis: BTC/ETH = qty x Binance spot price; other symbols = Bybit "
          "quote_qty (bankruptcy price, see T1.4)")

    pc = ctx.clusters[args.primary_gap]
    gate_rows = []
    for fl in args.floors_usd:
        n = int(((pc["symbol"].isin(["BTCUSDT", "ETHUSDT"])) & (pc["segment"] == "scheduled")
                 & (~pc["censored"]) & (pc["size_usd"] >= fl)).sum())
        gate_rows.append([_usd(fl), n, "VIABLE" if n >= args.t3_min_clusters else "TOO SMALL"])
    print(f"\n  T3 GATE -- BTC+ETH uncensored scheduled clusters at gap {args.primary_gap:g}s "
          f"(threshold {args.t3_min_clusters}):")
    print(fmt_table(gate_rows, ["floor", "clusters", "T3"]))

    print(banner("T1.4  BANKRUPTCY PRICE -- direction measured, not asserted"))
    bb = LI.bankruptcy_bias(ctx.sched, ctx.books)
    print(fmt_table([[r.asset, r.side, r.position_liquidated, r.subset, f"{r.n:,}",
                      _f(r.median_bps), _f(r.p25_bps), _f(r.p75_bps), _pct(r.frac_below_mid),
                      _f(r.median_dt_ms, 0), _f(r.bankruptcy_offset_bps),
                      _f(r.common_component_bps)] for r in bb.itertuples()],
                    ["asset", "side", "liquidated", "subset", "n", "med_bps", "p25", "p75",
                     "below_mid", "dt_ms", "offset", "common"]))
    print("  bias_bps = Bybit price vs Binance spot mid (nearest snapshot within 5s).\n"
          "  offset = half the between-side difference (isolates the bankruptcy offset);"
          "\n  common = mean of the sides (perp-spot basis + timing, same sign both sides).")
    _save(bb, out, "t1_bankruptcy_bias")

    print(banner("T1.5  OVERLAP WITH FLOW / DEPTH DATA"))
    for a in args.assets:
        cad = LI.book_cadence_in(ctx.books[a], cov.first_ms, ctx.end_ms)
        b = ctx.buckets[a]
        print(f"  {a}: trades buckets {ms_to_str(b['bucket_start_ms'].min())} -> "
              f"{ms_to_str(b['bucket_start_ms'].max())}; book in label window: "
              f"{cad['n']:,} snaps, median gap {_f(cad['median_gap_sec'], 1)}s, "
              f"{cad['gaps_over_30s']} gaps >30s totalling "
              f"{_f(cad['hours_in_gaps_over_30s'], 1)} h (max {_f(cad['max_gap_sec'] / 3600, 1)} h)")
    print("  SOL / XRP / ADA / AVAX: labels only -- no trades, no book, no 1m OHLCV "
          "in crypto_data.db.")
    return 0


# ================================================================== T2 =====

def cmd_t2(args, ctx: Ctx) -> int:
    out = ctx.out
    cov = ctx.cov
    print(banner("T2 -- THE CONFUSION MATRIX CYCLE 61 COULD NOT COMPUTE"))

    print("\nT2.0  REGRESSION CHECK (a control that can fail): Cycle 61 counts")
    old = {}
    for a in args.assets:
        p = C.bucket_cache_path(a, args.bucket_sec, args.c61_cache_dir)
        if p.exists():
            old[a] = pd.read_parquet(p)
    if old:
        rc = LS.regression_check(old, ctx.buckets, args.bucket_sec)
        print(fmt_table([[r.asset, r.setting, r.published, r.c61_cache_now_code,
                          r.c66_cache_truncated, "PASS" if r.code_unchanged else "FAIL",
                          "PASS" if r.data_reproduces else "FAIL", r.buckets_only_old,
                          r.buckets_only_new, r.buckets_value_mismatch]
                         for r in rc.itertuples()],
                        ["asset", "setting", "published", "c61cache+code", "c66cache_trunc",
                         "code", "data", "only_old", "only_new", "mismatch"]))
        _save(rc, out, "t2_regression_check")
    else:
        print("  Cycle 61 cache not found -- regression check NOT run")

    gap = args.primary_gap
    tau = args.primary_tau
    win_rows, d2l_rows, all_hits = [], [], []
    print(banner(f"T2.1  WINDOW-LEVEL 2x2 (W=60s detector windows inside observed time; "
                 f"cluster gap {gap:g}s, tau {tau:g}s)"))
    grid_rows = []
    for name, p in C.DEFAULT_SWEEP:
        for a in args.assets:
            d, ev = ctx.detection(a, name, p)
            for g in args.cluster_gaps:
                cl = ctx.asset_clusters(a, g)
                for fl in args.floors_usd:
                    for tt in args.taus_sec:
                        boot = args.n_boot if (g == gap and tt == tau) else 0
                        r = LS.window_confusion(d, p.window_sec, cl, fl, tt, cov.lo, cov.hi,
                                                n_boot=boot, seed=args.seed)
                        r.update(asset=a, setting=name, gap_sec=g)
                        grid_rows.append(r)
                        if g == gap and tt == tau:
                            win_rows.append(r)
    wdf = pd.DataFrame(grid_rows)
    _save(wdf, out, "t2_window_confusion_grid")
    print(fmt_table([[r["asset"], r["setting"], _usd(r["floor_usd"]), r["TP"], r["FP"],
                      r["FN"], f"{r['TN']:,}", _pct(r["precision"]),
                      "[" + ", ".join(_pct(x) for x in r.get("precision_ci90", [np.nan] * 2)) + "]",
                      _pct(r["prevalence"], 2), _f(r["lift"]),
                      "[" + ", ".join(_f(x) for x in r.get("lift_ci90", [np.nan] * 2)) + "]",
                      _pct(r["recall"]), _pct(r["fpr"], 3)] for r in win_rows],
                    ["asset", "setting", "floor", "TP", "FP", "FN", "TN", "precision",
                     "prec_ci90", "prevalence", "lift", "lift_ci90", "recall", "FPR"]))
    print("  prevalence = chance precision (a detector firing at random with the same rate);"
          " CIs resample whole days.")

    print(banner(f"T2.2  DETECTOR -> LABEL (each detector event; tau {tau:g}s)"))
    for name, p in C.DEFAULT_SWEEP:
        for a in args.assets:
            d, ev = ctx.detection(a, name, p)
            lab_ev = ev[(ev["start_ms"] >= cov.first_ms) & (ev["start_ms"] < ctx.end_ms)]
            for fl in args.floors_usd:
                r = LS.detector_to_label(lab_ev, ctx.asset_clusters(a, gap), fl, tau,
                                         cov.lo, cov.hi, seed=args.seed)
                r.update(asset=a, setting=name)
                d2l_rows.append(r)
    ddf = pd.DataFrame(d2l_rows)
    _save(ddf, out, "t2_detector_to_label")
    print(fmt_table([[r.asset, r.setting, _usd(r.floor_usd), r.events_in_mask,
                      r.events_outside_mask, r.matched, _pct(r.precision), _pct(r.chance),
                      _f(r.lift), _pct(r.side_agreement)] for r in ddf.itertuples()],
                    ["asset", "setting", "floor", "events", "unobserved", "matched",
                     "precision", "chance", "lift", "side_agree"]))

    print(banner("T2.3  LABEL -> DETECTOR: RECALL BY CLUSTER SIZE (pooled BTC+ETH)"))
    print("  OVERLAP = detector fired anywhere inside the cluster span;"
          f" STRICT = a detector event started within {tau:g}s of the cluster onset.")
    print("  null = same spans re-placed at random observed times (chance recall).")
    rec_rows = []
    for name, p in C.DEFAULT_SWEEP:
        hits_all, nulls_all = [], []
        man_hits = []
        for a in args.assets:
            d, ev = ctx.detection(a, name, p)
            cl = ctx.asset_clusters(a, gap)
            sc = cl[(cl["segment"] == "scheduled") & (~cl["censored"])]
            h = LS.label_to_detector(sc, d, ev, p.window_sec, tau)
            nl = LS.null_hit_rates(sc, d, ev, p.window_sec, tau, cov.lo, cov.hi,
                                   k=args.null_k, seed=args.seed)
            hits_all.append(h)
            nulls_all.append(nl)
            mc = cl[cl["segment"] == "manual"]
            if len(mc):
                man_hits.append(LS.label_to_detector(mc, d, ev, p.window_sec, tau))
        H = pd.concat(hits_all, ignore_index=True)
        N = pd.concat(nulls_all, ignore_index=True)
        H["setting"] = name
        all_hits.append(H.assign(null_overlap=N["null_overlap"].values,
                                 null_strict=N["null_strict"].values))
        for by in ("size_bucket", "count_bucket"):
            t = LS.recall_table(H, N, by)
            t.insert(0, "setting", name)
            t.insert(1, "by", by)
            t["manual"] = "without"
            rec_rows.append(t)
        if man_hits:
            M = pd.concat(man_hits, ignore_index=True)
            for _, r in M.iterrows():
                rec_rows.append(pd.DataFrame([{
                    "setting": name, "by": "MANUAL_62A", "size_bucket": r["size_bucket"],
                    "n": 1, "recall_overlap": float(r["hit_overlap"]),
                    "recall_strict": float(r["hit_strict"]), "manual": "with",
                    "median_duration_sec": float(r["duration_sec"]),
                    "note": f"{r['symbol']} {r['start_utc']} {_usd(r['size_usd'])}"}]))
    rdf = pd.concat(rec_rows, ignore_index=True)
    _save(rdf, out, "t2_recall_by_size")
    hits_df = pd.concat(all_hits, ignore_index=True)
    _save_rows(hits_df, args, "t2_cluster_hits")
    sz = rdf[(rdf["by"] == "size_bucket")]
    print(fmt_table([[r.setting, r.size_bucket, r.n, _pct(r.recall_overlap),
                      _pct(r.null_overlap), _f(r.lift_overlap), _pct(r.recall_strict),
                      _pct(r.null_strict), _f(r.lift_strict), _f(r.median_duration_sec, 0)]
                     for r in sz.itertuples()],
                    ["setting", "size", "n", "recall_ovl", "null_ovl", "lift_ovl",
                     "recall_str", "null_str", "lift_str", "dur_p50_s"]))
    man = rdf[rdf["by"] == "MANUAL_62A"]
    if len(man):
        print("\n  WITH the manual 62A cascade (event only; reported per cluster):")
        print(fmt_table([[r.setting, r.note, r.size_bucket,
                          "FIRED" if r.recall_overlap else "no",
                          "FIRED" if r.recall_strict else "no", _f(r.median_duration_sec, 0)]
                         for r in man.itertuples()],
                        ["setting", "cluster", "size", "overlap", "strict", "dur_s"]))
    cb = rdf[(rdf["by"] == "count_bucket") & (rdf["setting"] == "base")]
    print("\n  base setting, by EVENT COUNT:")
    print(fmt_table([[r.count_bucket, r.n, _pct(r.recall_overlap), _pct(r.null_overlap),
                      _f(r.lift_overlap), _pct(r.recall_strict), _pct(r.null_strict)]
                     for r in cb.itertuples()],
                    ["events", "n", "recall_ovl", "null_ovl", "lift_ovl", "recall_str",
                     "null_str"]))

    print(banner("T2.4  WHAT THE DETECTOR SAW -- its own statistics inside clusters"))
    diag_rows, top_all = [], []
    for a in args.assets:
        d, ev = ctx.detection(a, "base", C.sweep_params("base"))
        cl = ctx.asset_clusters(a, gap)
        sc = cl[(cl["segment"] == "scheduled") & (~cl["censored"])
                & (cl["size_usd"] >= args.floors_usd[0])]
        mc = cl[cl["segment"] == "manual"]
        dg = LS.cluster_diagnostics(pd.concat([sc, mc]), d, 60)
        dg["asset"] = a
        top_all.append(dg)
        durs = (sc["end_ms"] - sc["start_ms"]).values
        nd = LS.null_diagnostics(durs, d, 60, cov.lo, cov.hi, n=4000, seed=args.seed)
        for b in list(LI.SIZE_LABELS[1:]):
            g = dg[(dg["size_bucket"] == b) & (dg["segment"] == "scheduled")]
            if not len(g):
                continue
            diag_rows.append({
                "asset": a, "size": b, "n": len(g),
                "burst_p50": g["max_burst_ratio"].median(),
                "null_burst_p50": nd["max_burst_ratio"].median(),
                "imb_p50": g["max_abs_imbalance"].median(),
                "null_imb_p50": nd["max_abs_imbalance"].median(),
                "movez_p50": g["max_abs_move_z"].median(),
                "null_movez_p50": nd["max_abs_move_z"].median(),
                "K_any": g["base_anyK"].mean(), "I_any": g["base_anyI"].mean(),
                "M_any": g["base_anyM"].mean(), "all3": g["base_all3"].mean(),
                "KM_same": g["base_KM_same"].mean(),
                "imb_sign_agrees": g["imb_sign_at_max_burst_agrees"].mean()})
    dgdf = pd.DataFrame(diag_rows)
    print("  max over the cluster's 60s windows (base K/I/M = 5 / 0.8 / 3); null = same"
          " durations at random observed times")
    print(fmt_table([[r.asset, r.size, r.n, _f(r.burst_p50), _f(r.null_burst_p50),
                      _f(r.imb_p50), _f(r.null_imb_p50), _f(r.movez_p50),
                      _f(r.null_movez_p50), _pct(r.K_any), _pct(r.I_any), _pct(r.M_any),
                      _pct(r.all3), _pct(r.KM_same), _pct(r.imb_sign_agrees)]
                     for r in dgdf.itertuples()],
                    ["asset", "size", "n", "burst", "null", "|imb|", "null", "|move_z|",
                     "null", "K>5", "I>.8", "M>3", "all3", "K&M_1win", "imb_sign_ok"]))
    print("  K&M_1win is a DIAGNOSTIC of which condition blocked the as-built rule (I alone"
          " when K&M hold in one window).\n  It is not a detector and is never scored.")
    _save(dgdf, out, "t2_detector_statistics_in_clusters")
    top = pd.concat(top_all, ignore_index=True).sort_values("size_usd", ascending=False)
    topn = top.head(args.top_n)
    print(f"\n  The {args.top_n} largest BTC/ETH clusters (gap {gap:g}s), as the base detector saw them:")
    print(fmt_table([[r.asset, r.start_utc, r.segment[:5], _f(r.duration_sec, 0), r.n_events,
                      _usd(r.size_usd), "SELL" if r.direction < 0 else "BUY",
                      _f(r.max_burst_ratio), _f(r.max_abs_imbalance), _f(r.max_abs_move_z),
                      "".join(k for k, f in (("K", r.base_anyK), ("I", r.base_anyI),
                                             ("M", r.base_anyM)) if f) or "-",
                      "FIRED" if r.base_all3 else "no",
                      "FIRED" if r.loose_all3 else "no"] for r in topn.itertuples()],
                    ["asset", "start", "seg", "dur_s", "n", "size", "forced", "burst",
                     "|imb|", "|mz|", "conds_met", "base", "loose"]))
    _save_rows(top, args, "t2_cluster_diagnostics")
    _save(top.head(100), out, "t2_top100_cluster_diagnostics")

    print(banner("T2.5  WINDOW SWEEP (Cycle 61's own W sweep at base K/I/M): overlap recall"))
    b = C.sweep_params("base")
    ws_rows = []
    for w in args.window_sweep:
        L = max(int(round(86400 / w)), 30)
        p = C.CascadeParams(window_sec=w, lookback_windows=L, burst_k=b.burst_k,
                            imbalance_i=b.imbalance_i, move_m=b.move_m)
        H, N = [], []
        for a in args.assets:
            d, ev = ctx.detection(a, f"W{w}", p)
            cl = ctx.asset_clusters(a, gap)
            sc = cl[(cl["segment"] == "scheduled") & (~cl["censored"])]
            H.append(LS.label_to_detector(sc, d, ev, w, tau))
            N.append(LS.null_hit_rates(sc, d, ev, w, tau, cov.lo, cov.hi, k=args.null_k,
                                       seed=args.seed))
            n_lab = int(((ev["start_ms"] >= cov.first_ms) & (ev["start_ms"] < ctx.end_ms)).sum())
            ws_rows.append({"W": w, "asset": a, "events_in_label_window": n_lab})
        t = LS.recall_table(pd.concat(H, ignore_index=True), pd.concat(N, ignore_index=True),
                            "size_bucket")
        for r in t.itertuples():
            ws_rows.append({"W": w, "size": r.size_bucket, "n": r.n,
                            "recall_overlap": getattr(r, "recall_overlap", np.nan),
                            "null_overlap": getattr(r, "null_overlap", np.nan),
                            "lift_overlap": getattr(r, "lift_overlap", np.nan)})
    wsdf = pd.DataFrame(ws_rows)
    _save(wsdf, out, "t2_window_sweep_recall")
    ev_counts = wsdf[wsdf["asset"].notna()]
    print("  detector events inside the label window: "
          + ", ".join(f"W={int(r.W)}s {r.asset}={int(r.events_in_label_window)}"
                      for r in ev_counts.itertuples()))
    rr = wsdf[wsdf["size"].notna()]
    print(fmt_table([[f"{int(r.W)}s", r.size, int(r.n), _pct(r.recall_overlap),
                      _pct(r.null_overlap), _f(r.lift_overlap)] for r in rr.itertuples()],
                    ["W", "size", "n", "recall_ovl", "null_ovl", "lift"]))

    print(banner("T2.6  ORDER-BOOK STRESS AT LABELLED CLUSTERS (vs Cycle 61's depth 1.02)"))
    ex_lo, ex_hi = ctx.exclusion_spans()
    st_rows = []
    for a in args.assets:
        book = ctx.books[a]
        pool = LS.control_pool(cov.lo, cov.hi, ex_lo, ex_hi, step_ms=60_000,
                               span_ms=int(args.stress_min_dur_sec * 1000))
        vol = LS.trailing_vol(ctx.grids[a], args.vol_lookback_min)
        cl = ctx.asset_clusters(a, gap)
        sets = []
        for bkt in LI.SIZE_LABELS[1:]:
            sel = cl[(cl["segment"] == "scheduled") & (~cl["censored"])
                     & (cl["size_bucket"] == bkt)]
            sets.append((f"labels {bkt}", sel))
        sets.append(("labels >=100k (all)", cl[(cl["segment"] == "scheduled") & (~cl["censored"])
                                               & (cl["size_usd"] >= 1e5)]))
        man = cl[cl["segment"] == "manual"]
        if len(man):
            sets.append(("labels >=100k + MANUAL", pd.concat([
                cl[(cl["segment"] == "scheduled") & (~cl["censored"]) & (cl["size_usd"] >= 1e5)],
                man[man["size_usd"] >= 1e5]])))
        d, ev = ctx.detection(a, "base", C.sweep_params("base"))
        lab_ev = ev[LB.spans_inside(cov.lo, cov.hi, ev["start_ms"].values, ev["end_ms"].values)]
        lab_ev = lab_ev.assign(direction=lab_ev["dom_side"].values)
        sets.append(("detector base events", lab_ev))
        for label, sel in sets:
            if not len(sel):
                st_rows.append({"asset": a, "set": label, "n": 0})
                continue
            spans = LS.stress_spans(sel, args.stress_min_dur_sec)
            uni = LS.control_spans_for(spans, LS.uniform_controls(len(spans), pool, k=args.control_k,
                                                                    seed=args.seed))
            mc = LS.matched_controls(spans["start_ms"].values, pool, ctx.grids[a], vol,
                                     k=args.control_k, n_vol_buckets=args.vol_buckets,
                                     seed=args.seed)
            mat = LS.control_spans_for(spans, mc)
            su = LS.book_stress(spans, uni, book, args.baseline_hours)
            sm = LS.book_stress(spans, mat, book, args.baseline_hours)
            for metric in ("depth", "hit_side_depth", "spread_bps"):
                st_rows.append({
                    "asset": a, "set": label, "metric": metric, "n": su[metric].get("n_events"),
                    "event_median": su[metric].get("ev_median"),
                    "uniform_ctrl_median": su[metric].get("ctrl_median"),
                    "ev_over_uniform": su[metric].get("ev_over_ctrl"),
                    "matched_ctrl_median": sm[metric].get("ctrl_median"),
                    "ev_over_matched": sm[metric].get("ev_over_ctrl"),
                    "thin_cells": int(mc["thin_cell"].sum()) if len(mc) else 0})
    sdf = pd.DataFrame(st_rows)
    _save(sdf, out, "t2_book_stress")
    print(fmt_table([[r.asset, r.set, r.metric, r.n, _f(r.event_median, 3),
                      _f(r.uniform_ctrl_median, 3), _f(r.ev_over_uniform, 3),
                      _f(r.matched_ctrl_median, 3), _f(r.ev_over_matched, 3)]
                     for r in sdf.dropna(subset=["metric"]).itertuples()],
                    ["asset", "set", "metric", "n", "ev_med", "unif_ctrl", "ev/unif",
                     "match_ctrl", "ev/MATCHED"]))
    print("  ratios are each span's mean vs its own trailing 24h median (Cycle 61's"
          " definition); hit side = bid under\n  forced selling, ask under forced buying."
          " Controls: observed, cascade-free (>=$100k clusters +/-900s excluded).")
    return 0


# ================================================================== T3 =====

def cmd_t3(args, ctx: Ctx) -> int:
    out = ctx.out
    gap = args.primary_gap
    print(banner(f"T3 -- WHAT A LABELLED CASCADE LOOKS LIKE (gap {gap:g}s, "
                 f"floor {_usd(args.t3_floor_usd)})"))
    pc = ctx.clusters[gap]
    bt = pc[pc["symbol"].isin(["BTCUSDT", "ETHUSDT"])]
    ev_sched = bt[(bt["segment"] == "scheduled") & (~bt["censored"])
                  & (bt["size_usd"] >= args.t3_floor_usd)]
    n = len(ev_sched)
    print(f"\n  GATE: {n} BTC+ETH uncensored scheduled clusters >= {_usd(args.t3_floor_usd)} "
          f"(threshold {args.t3_min_clusters})")
    if n < args.t3_min_clusters:
        print("  SAMPLE TOO SMALL -- T3 stops here, as briefed.")
        _save({"gate": n, "threshold": args.t3_min_clusters, "ran": False}, out, "t3_gate")
        return 0
    man = bt[(bt["segment"] == "manual") & (bt["size_usd"] >= args.t3_floor_usd)]
    ex_lo, ex_hi = ctx.exclusion_spans()
    variants = [("without_manual", ev_sched)]
    if len(man):
        variants.append(("with_manual", pd.concat([ev_sched, man])))
    all_rev, all_prof = [], []
    for vname, evs in variants:
        rev_parts, prof_parts, pooled = [], [], []
        for a in args.assets:
            e = CP.event_frame(evs[evs["symbol"] == ASSET_SYMBOL[a]])
            if not len(e):
                continue
            series = CP.build_series(ctx.buckets[a], ctx.books[a])
            pool = LS.control_pool(ctx.cov.lo, ctx.cov.hi, ex_lo, ex_hi, step_ms=60_000,
                                   span_ms=60_000)
            vol = LS.trailing_vol(ctx.grids[a], args.vol_lookback_min)
            mc = LS.matched_controls(e["start_ms"].values, pool, ctx.grids[a], vol,
                                     k=args.control_k, n_vol_buckets=args.vol_buckets,
                                     seed=args.seed)
            idx = mc["event_idx"].values
            c_start = mc["anchor_ms"].values
            dur = (e["end_ms"] - e["start_ms"]).values[idx]
            c_dir = e["direction"].values[idx]
            ev_m = CP.moves(series, e["start_ms"].values, e["end_ms"].values,
                            e["direction"].values, gap, args.horizons_min)
            ct_m = CP.moves(series, c_start, c_start + dur, c_dir, gap, args.horizons_min)
            ev_p = CP.profiles(series, e["start_ms"].values, e["direction"].values,
                               baseline_hours=args.baseline_hours)
            ct_p = CP.profiles(series, c_start, c_dir, baseline_hours=args.baseline_hours)
            rt = CP.reversion_table(ev_m, ct_m, idx, e["day"].values, args.costs_bps)
            rt.insert(0, "asset", a)
            rt.insert(0, "variant", vname)
            rev_parts.append(rt)
            pooled.append((ev_m, ct_m, idx, e["day"].values))
            pt = CP.profile_table(ev_p, ct_p, idx)
            pt.insert(0, "asset", a)
            pt.insert(0, "variant", vname)
            prof_parts.append(pt)
            ev_m2 = ev_m.copy()
            ev_m2["asset"], ev_m2["variant"] = a, vname
            ev_m2["start_utc"] = e["start_utc"].values
            ev_m2["size_usd"] = e["size_usd"].values
            ev_m2["duration_sec"] = e["duration_sec"].values
            ev_m2["direction"] = e["direction"].values
            ev_m2["segment"] = e["segment"].values
            _save(ev_m2, out, f"t3_event_moves_{a}_{vname}")
            print(f"\n  [{vname}] {a}: {len(e)} events, {len(mc)} matched controls "
                  f"({int(mc['thin_cell'].sum())} from thin cells); size p50 "
                  f"{_usd(e['size_usd'].median())}, max {_usd(e['size_usd'].max())}; "
                  f"duration p50 {e['duration_sec'].median():.0f}s, p90 "
                  f"{e['duration_sec'].quantile(0.9):.0f}s; forced-sell share "
                  f"{(e['direction'] < 0).mean():.0%}")
        if len(pooled) > 1:
            # Assets pooled: bps are comparable units, and a shared UTC day is
            # one cluster for the day-block CI and the day-clustered SE.
            off = np.cumsum([0] + [len(p[0]) for p in pooled[:-1]])
            rt = CP.reversion_table(pd.concat([p[0] for p in pooled], ignore_index=True),
                                    pd.concat([p[1] for p in pooled], ignore_index=True),
                                    np.concatenate([p[2] + o for p, o in zip(pooled, off)]),
                                    np.concatenate([p[3] for p in pooled]), args.costs_bps)
            rt.insert(0, "asset", "+".join(args.assets))
            rt.insert(0, "variant", vname)
            rev_parts.append(rt)
        all_rev.extend(rev_parts)
        all_prof.extend(prof_parts)
    rev = pd.concat(all_rev, ignore_index=True)
    prof = pd.concat(all_prof, ignore_index=True)
    _save(rev, out, "t3_reversion")
    _save(prof, out, "t3_profiles")

    p0 = prof[prof["variant"] == "without_manual"]
    print(banner("T3.1  EVENT-TIME PROFILE vs MATCHED CONTROL (medians; oriented to the forced flow)"))
    for a in args.assets:
        pa = p0[p0["asset"] == a]
        if not len(pa):
            continue
        piv = {}
        for ch in ("price_bps", "flow_usd", "depth_ratio", "hit_depth_ratio", "spread_ratio"):
            q = pa[pa["channel"] == ch].set_index("offset_sec")
            piv[ch] = q
        offs = sorted(pa["offset_sec"].unique())
        print(f"\n  {a}: offsets from cluster START")
        rows = []
        for o in offs:
            r = [f"{o:+d}s"]
            for ch, nd in (("price_bps", 1), ("flow_usd", 0), ("depth_ratio", 3),
                           ("hit_depth_ratio", 3), ("spread_ratio", 3)):
                q = piv[ch].loc[o]
                val = (_usd(q["event_median"]) if ch == "flow_usd" else _f(q["event_median"], nd))
                cval = (_usd(q["control_median"]) if ch == "flow_usd"
                        else _f(q["control_median"], nd))
                r.append(f"{val} / {cval}")
            rows.append(r)
        print(fmt_table(rows, ["offset", "price bps ev/ctl", "flow ev/ctl",
                               "depth ev/ctl", "hit-depth ev/ctl", "spread ev/ctl"]))

    print(banner("T3.2  SIZE, MOVE AND REVERSION -- GROSS bps FIRST"))
    for vname in rev["variant"].unique():
        for a in rev.loc[rev["variant"] == vname, "asset"].unique():
            ra = rev[(rev["variant"] == vname) & (rev["asset"] == a)]
            if not len(ra):
                continue
            print(f"\n  [{vname}] {a}  (rev_* measured from end + {gap:g}s, the first "
                  f"knowable instant; *_end is look-ahead reference)")
            print(fmt_table([[r.metric, r.n, _f(r.event_mean_bps), _f(r.event_median_bps),
                              _f(r.control_mean_bps), _f(r.excess_mean_bps),
                              _f(r.excess_sd_bps, 1), _f(r.excess_se_day_bps),
                              f"[{_f(r.excess_ci90_lo)}, {_f(r.excess_ci90_hi)}]",
                              _pct(r.event_hit_rate)] for r in ra.itertuples()],
                            ["metric", "n", "ev_mean", "ev_med", "ctl_mean", "EXCESS", "sd",
                             "se_day", "excess_ci90", "ev>0"]))
    print(banner("T3.3  NET OF A ROUND-TRIP COST GRID (event mean, knowable entry)"))
    cols = [c for c in rev.columns if c.startswith("net_") and c.endswith("_mean")]
    for vname in rev["variant"].unique():
        ra = rev[(rev["variant"] == vname) & rev["metric"].str.match(r"^rev_\d+m$")]
        print(f"\n  [{vname}]  costs: " + "; ".join(f"{c:g}bp = {CP.COST_LABELS.get(c, '')}"
                                                   for c in args.costs_bps))
        print(fmt_table([[r["asset"], r["metric"], _f(r["event_mean_bps"])]
                         + [_f(r[c]) for c in cols] for _, r in ra.iterrows()],
                        ["asset", "horizon", "gross"] + [c.replace("_mean", "") for c in cols]))
    return 0


# ================================================================== T4 =====

def cmd_t4(args, ctx: Ctx) -> int:
    out = ctx.out
    gap = args.primary_gap
    print(banner("T4 -- SIZING THE PATH-SIGNATURE QUESTION (no implementation)"))
    pc = ctx.clusters[gap]
    bt = pc[pc["symbol"].isin(["BTCUSDT", "ETHUSDT"]) & (pc["segment"] == "scheduled")
            & (~pc["censored"])]
    samples = {f">={_usd(fl)}": int((bt["size_usd"] >= fl).sum()) for fl in args.floors_usd}
    print("\n  Supervised sample (BTC+ETH labelled clusters, gap "
          f"{gap:g}s, uncensored): " + ", ".join(f"{k}: {v}" for k, v in samples.items()))
    tab = SS.sizing_table(samples)
    _save(tab, out, "t4_term_counts")
    print(fmt_table([[r["channels"], r["channel_set"], r["level"], r["sig_terms"],
                      r["logsig_terms"]] + [_f(r[f"sig_per_label[{k}]"], 1) for k in samples]
                     for _, r in tab.iterrows()],
                    ["d", "channels", "level", "sig", "logsig"]
                    + [f"sig/label {k}" for k in samples]))
    print("  sig/label >= 10 means features outnumber labelled clusters by an order of"
          " magnitude.")
    if args.tooling_results and Path(args.tooling_results).exists():
        tr = json.loads(Path(args.tooling_results).read_text(encoding="utf-8"))
        _save(tr, out, "t4_tooling_results")
        rows = []
        for k, v in tr.items():
            if k.startswith("_"):
                continue
            s = v.get("smoke", {}) or {}
            rows.append([k, v.get("pip_rc"), s.get("version"), s.get("computes"),
                         s.get("matches_reference"), (s.get("error") or "")[:60]])
        print("\n  Tooling (throwaway venvs from the 3.12.10 base interpreter; .venv untouched):")
        print(fmt_table(rows, ["package", "pip_rc", "version", "computes", "matches_ref",
                               "error"]))
        for k, v in tr.get("_pass2", {}).items():
            print(f"    {k:12s} {v}")
        for k, v in tr.get("_environment", {}).items():
            print(f"    env {k}: {v}")
    return 0


# ============================================================ DB commands ==

def cmd_extract(args) -> int:
    time_guard(args)
    print(banner("EXTRACT -- read-only snapshot of crypto_data.db inputs"))
    for a in args.assets:
        C.build_bucket_cache(a, bucket_sec=args.bucket_sec, db_path=args.db,
                             cache_dir=args.cache_dir, force=False)
    m = LB.extract_inputs(args.db, args.cache_dir, tuple(args.assets), args.log_path,
                          force=args.refresh_inputs)
    print(json.dumps({k: v for k, v in m.items() if k != "queries"}, indent=1, default=str))
    return 0


def cmd_contiguity(args, ctx: Ctx) -> int:
    """Is the spot tape complete where it matters? Binance trade ids are
    sequential, so inside any window count == max_id - min_id + 1 unless the
    collector dropped trades. Checked around every BTC/ETH cluster >= the T3
    floor (incl. manual) and around random observed control windows."""
    time_guard(args)
    print(banner("CONTIGUITY -- trade_id completeness around labelled clusters (DB, mode=ro)"))
    pc = ctx.clusters[args.primary_gap]
    rows = []
    rng = np.random.default_rng(args.seed)
    for a in args.assets:
        cl = pc[(pc["symbol"] == ASSET_SYMBOL[a]) & (pc["size_usd"] >= args.t3_floor_usd)]
        spans = [(int(s) - 60_000, int(e) + 60_000, "cluster") for s, e in
                 zip(cl["start_ms"], cl["end_ms"])]
        w = (ctx.cov.hi - ctx.cov.lo).astype(float)
        k = rng.choice(len(w), size=args.contiguity_controls, p=w / w.sum())
        t = ctx.cov.lo[k] + (rng.random(len(k)) * np.maximum(w[k] - 180_000, 1)).astype("int64")
        spans += [(int(x), int(x) + 180_000, "control") for x in t]
        with read_only_db(args.db) as conn:
            for lo, hi, kind in spans:
                n, mn, mx = conn.execute(
                    "SELECT COUNT(*), MIN(trade_id), MAX(trade_id) FROM trades "
                    "WHERE asset=? AND timestamp >= ? AND timestamp < ?", (a, lo, hi)).fetchone()
                rows.append({"asset": a, "kind": kind, "start_utc": ms_to_str(lo),
                             "trades": n, "id_span": (mx - mn + 1) if n else 0,
                             "missing": ((mx - mn + 1) - n) if n else None})
    df = pd.DataFrame(rows)
    _save(df, ctx.out, "contiguity_trade_ids")
    summ = df.groupby(["asset", "kind"]).agg(windows=("trades", "size"),
                                             trades=("trades", "sum"),
                                             missing=("missing", "sum"),
                                             windows_with_missing=("missing",
                                                                   lambda s: int((s > 0).sum())))
    print(summ.to_string())
    return 0


# ================================================================== CLI ====

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cycle 66 A2 validation against labelled liquidations")
    p.add_argument("-v", "--verbose", action="count", default=2,
                   help="-v INFO, -vv DEBUG (default: maximum)")
    p.add_argument("-q", "--quiet", action="store_true")
    p.add_argument("--validate", dest="validate", action="store_true", default=True)
    p.add_argument("--no-validate", dest="validate", action="store_false")
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--cache-dir", default=str(LB.C66_CACHE_DIR))
    p.add_argument("--out-dir", default=str(LB.C66_OUT_DIR))
    p.add_argument("--c61-cache-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--log-path", default=str(LB.DEFAULT_COLLECTOR_LOG))
    p.add_argument("--log-tz", default=LB.LOG_TZ)
    p.add_argument("--assets", nargs="+", default=["BTC", "ETH"])
    p.add_argument("--bucket-sec", type=int, default=10)
    p.add_argument("--seed", type=int, default=66)
    p.add_argument("--refresh-inputs", action="store_true",
                   help="re-extract the snapshot (overwrites this cycle's own cache files)")
    p.add_argument("--guard-window", default="02:50-03:45",
                   help="local HH:MM-HH:MM during which DB commands refuse to run")
    p.add_argument("--ignore-time-guard", action="store_true")
    # coverage mask
    p.add_argument("--connect-pad-ms", type=int, default=1000)
    p.add_argument("--disconnect-pad-ms", type=int, default=60_000)
    p.add_argument("--band-pad-sec", type=float, default=2.0)
    p.add_argument("--no-band", dest="apply_band", action="store_false", default=True)
    p.add_argument("--band-extra-pre-sec", type=float, default=0.0,
                   help="SENSITIVITY: start the hourly band this much earlier")
    p.add_argument("--band-soh", type=float, nargs=2, default=None, metavar=("START", "END"),
                   help="SENSITIVITY: replace the measured band with [START, END) seconds "
                        "of the hour, e.g. 3540 30 for :59:00-:00:30 (the bounding rule)")
    p.add_argument("--local-baseline-secs", type=int, nargs="+", default=[60, 120, 300],
                   help="adjacent-window widths for the handoff local-baseline test")
    p.add_argument("--handoff-boot", type=int, default=1000)
    p.add_argument("--rows-dir", default=None,
                   help="where row-level parquet goes (default <cache-dir>/rows)")
    p.add_argument("--edge-tol-ms", type=int, default=2000)
    # clusters
    p.add_argument("--cluster-gaps", type=float, nargs="+", default=[10.0, 30.0, 60.0])
    p.add_argument("--primary-gap", type=float, default=30.0)
    p.add_argument("--floors-usd", type=float, nargs="+", default=[1e5, 1e6, 1e7])
    # scoring
    p.add_argument("--taus-sec", type=float, nargs="+", default=[0.0, 60.0, 300.0])
    p.add_argument("--primary-tau", type=float, default=60.0)
    p.add_argument("--window-sweep", type=int, nargs="+", default=[30, 60, 300, 900])
    p.add_argument("--null-k", type=int, default=20)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--top-n", type=int, default=15)
    p.add_argument("--exclusion-sec", type=int, default=900)
    p.add_argument("--exclusion-floor-usd", type=float, default=1e5)
    p.add_argument("--control-k", type=int, default=20)
    p.add_argument("--vol-lookback-min", type=int, default=60)
    p.add_argument("--vol-buckets", type=int, default=5)
    p.add_argument("--baseline-hours", type=float, default=24.0)
    p.add_argument("--stress-min-dur-sec", type=float, default=60.0)
    # T3
    p.add_argument("--t3-floor-usd", type=float, default=1e6)
    p.add_argument("--t3-min-clusters", type=int, default=50)
    p.add_argument("--horizons-min", type=int, nargs="+", default=list(CP.HORIZONS_MIN))
    p.add_argument("--costs-bps", type=float, nargs="+", default=list(CP.COSTS_BPS))
    # T4 / contiguity
    p.add_argument("--tooling-results", default=None)
    p.add_argument("--contiguity-controls", type=int, default=200)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("extract", "t1-inventory", "t2-confusion", "t3-profile", "t4-sizing",
                 "contiguity", "all"):
        sub.add_parser(name)
    return p


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    setup_logging(0 if args.quiet else args.verbose)
    if args.primary_gap not in args.cluster_gaps:
        args.cluster_gaps = sorted(set(args.cluster_gaps) | {args.primary_gap})
    t0 = time.time()
    if args.cmd == "extract":
        return cmd_extract(args)
    ctx = Ctx(args)
    rc = 0
    if args.cmd in ("t1-inventory", "all"):
        rc |= cmd_t1(args, ctx)
    if args.cmd in ("t2-confusion", "all"):
        rc |= cmd_t2(args, ctx)
    if args.cmd in ("t3-profile", "all"):
        rc |= cmd_t3(args, ctx)
    if args.cmd in ("t4-sizing", "all"):
        rc |= cmd_t4(args, ctx)
    if args.cmd == "contiguity":
        rc |= cmd_contiguity(args, ctx)
    print(f"\n[done] {args.cmd} in {time.time() - t0:.0f}s")
    return rc


if __name__ == "__main__":
    sys.exit(main())
