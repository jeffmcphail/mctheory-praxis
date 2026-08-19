#!/usr/bin/env python
"""
engines/forced_trade/run_audit.py

Cycle 61 forced-trade DATA AUDIT -- command line entry point.

    python -m engines.forced_trade.run_audit all
    python -m engines.forced_trade.run_audit t1-cascades
    python -m engines.forced_trade.run_audit t2-unlocks
    python -m engines.forced_trade.run_audit t3-leveraged
    python -m engines.forced_trade.run_audit t4-oi
    python -m engines.forced_trade.run_audit t5-occupancy

MEASUREMENT ONLY. No strategy, no P&L, no Sharpe, no backtest. Counts,
detectability, feasibility.

Rule 25: --validate and -vv are the DEFAULTS. Use --no-validate / -q to relax.
The database is opened read-only at the driver level (see common.read_only_db),
so the read-only requirement is enforced rather than promised.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from engines.forced_trade import cascade as C
from engines.forced_trade import corroborate as CO
from engines.forced_trade import leveraged as LV
from engines.forced_trade import occupancy as OC
from engines.forced_trade import oi_audit as OI
from engines.forced_trade import unlocks as UN
from engines.forced_trade.common import (
    DEFAULT_CACHE_DIR, DEFAULT_DB, DEFAULT_OUT_DIR, asset_spans, banner,
    ensure_dir, fmt_table, ms_to_str, per_year, read_only_db, setup_logging,
)

logger = logging.getLogger("forced_trade.run")


# ------------------------------------------------------------------ utils --

def _save(df_or_obj, out_dir: Path, name: str) -> Path:
    ensure_dir(out_dir)
    if isinstance(df_or_obj, pd.DataFrame):
        p = out_dir / f"{name}.csv"
        df_or_obj.to_csv(p, index=False)
    else:
        p = out_dir / f"{name}.json"
        p.write_text(json.dumps(df_or_obj, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", p)
    return p


def _pct(x, nd=1):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100*x:.{nd}f}%"


def _f(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


# ================================================================ T1 ======

def cmd_t1(args) -> int:
    out_dir = Path(args.out_dir)
    print(banner("T1 -- A2 LIQUIDATION CASCADES: identifiable without a feed?"))

    with read_only_db(args.db) as conn:
        spans = asset_spans(conn, "trades")
    print("\nUSABLE SPAN (trades table -- this CAPS the whole study):")
    for s in spans:
        print(f"  {s.asset}: {ms_to_str(s.first_ms)} -> {ms_to_str(s.last_ms)}  "
              f"({s.days:.1f} days)")
    span_days = max(s.days for s in spans)
    print(f"\n  ==> {span_days:.1f} days of tick data on {len(spans)} assets "
          f"(BTC, ETH only). Every per-year figure below is a "
          f"{365.25/span_days:.1f}x extrapolation from that window and is "
          f"reported next to the raw count, never instead of it.")

    if args.validate:
        print(f"\n[validate] side <-> is_buyer_maker "
              f"({'FULL SCAN' if args.validate_full_scan else 'sampled'}):")
        conv = C.validate_side_convention(
            args.db, tuple(a.asset for a in spans),
            full_scan=args.validate_full_scan, sample_days=args.validate_sample_days)
        for a, v in conv.items():
            print(f"  {a}: {v['violations']} violations in {v['rows_checked']:,} rows "
                  f"[{v['scope']}] -> {'PASS' if v['consistent'] else 'FAIL'} "
                  f"({v['elapsed_sec']}s)")
        _save(conv, out_dir, "t1_side_convention")

    assets = [s.asset for s in spans]
    buckets, results, all_events = {}, [], {}
    for a in assets:
        buckets[a] = C.build_bucket_cache(a, bucket_sec=args.bucket_sec,
                                          db_path=args.db, cache_dir=args.cache_dir,
                                          force=args.rebuild_cache)
        if args.validate:
            v = C.validate_bucket_cache(a, buckets[a], args.bucket_sec,
                                        n_sample=args.validate_sample, db_path=args.db)
            print(f"\n[validate] {a} bucket cache: {v['checked']} buckets re-derived "
                  f"from source, price_mismatch={v['price_mismatch']} "
                  f"flow_mismatch={v['flow_mismatch']} -> "
                  f"{'PASS' if v['ok'] else 'FAIL'}")
            if not v["ok"]:
                print(f"  MISMATCH EXAMPLES: {v['examples']}")
            _save(v, out_dir, f"t1_cache_validation_{a}")

    # ---- threshold sensitivity ------------------------------------------
    print(banner("T1.1  THRESHOLD SENSITIVITY (the spread IS the finding)"))
    rows = []
    for name, p in C.DEFAULT_SWEEP:
        for a in assets:
            r = C.run_detector(a, buckets[a], args.bucket_sec, name, p)
            results.append(r)
            all_events[(a, name)] = r.events
            dur = r.events["duration_sec"]
            rows.append([
                name, a, p.label().split(" ", 1)[1], r.n_events,
                f"{r.events_per_year:.0f}",
                f"{r.n_candidate_windows:,}",
                f"{100.0*r.n_candidate_windows/max(r.n_windows,1):.3f}%",
                f"{dur.median():.0f}" if len(dur) else "-",
                f"{dur.max():.0f}" if len(dur) else "-",
                _pct(r.events["dir_agree_frac"].mean()) if len(r.events) else "-",
            ])
    print()
    print(fmt_table(rows, ["setting", "asset", "params", "events", "ev/yr",
                           "cand_win", "cand_rate", "med_dur_s", "max_dur_s",
                           "dir_agree"]))
    _save(pd.DataFrame(rows, columns=["setting", "asset", "params", "events",
                                      "events_per_year", "candidate_windows",
                                      "candidate_rate", "median_duration_sec",
                                      "max_duration_sec", "mean_dir_agree"]),
          out_dir, "t1_threshold_sensitivity")

    # ---- window-size sensitivity ----------------------------------------
    print(banner("T1.2  WINDOW-SIZE SENSITIVITY (W held against fixed K/I/M)"))
    wrows = []
    b = C.sweep_params("base")   # single source of truth for K/I/M
    for w in args.window_sweep:
        L = max(int(round(86400 / w)), 30)   # keep the trailing window at ~1 day
        p = C.CascadeParams(window_sec=w, lookback_windows=L,
                            burst_k=b.burst_k, imbalance_i=b.imbalance_i,
                            move_m=b.move_m)
        for a in assets:
            r = C.run_detector(a, buckets[a], args.bucket_sec, f"W{w}", p)
            dur = r.events["duration_sec"]
            wrows.append([f"{w}s", a, r.n_events, f"{r.events_per_year:.0f}",
                          f"{dur.median():.0f}" if len(dur) else "-",
                          _pct(r.events["dir_agree_frac"].mean()) if len(r.events) else "-"])
    print()
    print(fmt_table(wrows, ["W", "asset", "events", "ev/yr", "med_dur_s", "dir_agree"]))
    _save(pd.DataFrame(wrows, columns=["window", "asset", "events",
                                       "events_per_year", "median_duration_sec",
                                       "mean_dir_agree"]),
          out_dir, "t1_window_sensitivity")

    # ---- duration distribution ------------------------------------------
    print(banner("T1.3  EVENT DURATION DISTRIBUTION (base setting)"))
    drows = []
    for a in assets:
        ev = all_events[(a, "base")]
        if ev.empty:
            drows.append([a, 0] + ["-"] * 6)
            continue
        d = ev["duration_sec"]
        drows.append([a, len(d)] + [f"{d.quantile(q):.0f}" for q in
                                    (0.10, 0.25, 0.50, 0.75, 0.90)] + [f"{d.max():.0f}"])
    print()
    print(fmt_table(drows, ["asset", "n", "p10", "p25", "median", "p75", "p90", "max"]))

    # ---- book corroboration ---------------------------------------------
    print(banner("T1.4  ORDER-BOOK CORROBORATION (depth collapse vs random control)"))
    book_rows, cad_rows = [], []
    for a in assets:
        book = CO.load_book(a, db_path=args.db)
        cad = CO.book_cadence(book)
        cad_rows.append([a, f"{cad.get('n',0):,}", _f(cad.get("median_gap_sec")),
                         _f(cad.get("p90_gap_sec")), _f(cad.get("p99_gap_sec")),
                         _f(cad.get("max_gap_sec"), 0),
                         _pct(cad.get("frac_gaps_over_60s"), 2)])
        for name, _ in C.DEFAULT_SWEEP:
            ev = all_events[(a, name)]
            prof = CO.depth_profile(ev, book, baseline_hours=args.baseline_hours)
            r0 = [r for r in results if r.asset == a and r.setting == name][0]
            ctrl = CO.random_control(ev, book, r0.first_ms, r0.last_ms,
                                     n_draws=args.control_draws,
                                     baseline_hours=args.baseline_hours,
                                     seed=args.seed)
            for metric, col in (("depth", "depth_ratio"), ("spread_bps", "spread_ratio")):
                s = CO.summarise_ratio(metric, prof.get(col, pd.Series(dtype=float)),
                                       ctrl.get(col, pd.Series(dtype=float)))
                book_rows.append([
                    a, name, metric, s.get("n_events", 0),
                    _f(s.get("ev_median")), _f(s.get("ctrl_median")),
                    _f(s.get("ev_over_ctrl")), _pct(s.get("ev_frac_below_1")),
                    _pct(s.get("ctrl_frac_below_1"))])
    print("\nSnapshot cadence (sets the corroboration resolution floor):")
    print(fmt_table(cad_rows, ["asset", "snaps", "med_gap_s", "p90_s", "p99_s",
                               "max_s", "gaps>60s"]))
    print("\nRatio vs own 24h trailing baseline, at events vs at matched random times:")
    print(fmt_table(book_rows, ["asset", "setting", "metric", "n", "ev_med",
                                "ctrl_med", "ev/ctrl", "ev<1", "ctrl<1"]))
    _save(pd.DataFrame(book_rows, columns=["asset", "setting", "metric", "n_events",
                                           "event_median", "control_median",
                                           "event_over_control", "event_frac_below_1",
                                           "control_frac_below_1"]),
          out_dir, "t1_book_corroboration")

    # ---- false-positive assessment --------------------------------------
    print(banner("T1.5  FALSE-POSITIVE ASSESSMENT (honest: no feed, no label)"))
    print("""
There is NO liquidation feed, so no detection can be labelled true or false and
NO FALSE-POSITIVE RATE IS COMPUTABLE. What follows is circumstantial evidence,
labelled as such. A forced liquidation and a large discretionary market order
are identical in the tape.""")
    conc_rows, xa_rows, ex_rows = [], [], []
    for name, _ in C.DEFAULT_SWEEP:
        for a in assets:
            r0 = [r for r in results if r.asset == a and r.setting == name][0]
            cnt = CO.daily_counts(all_events[(a, name)], r0.first_ms, r0.last_ms)
            c = CO.concentration(cnt)
            conc_rows.append([a, name, c.get("n_events", 0), c.get("days_with_event", 0),
                              c.get("n_days", 0), _f(c.get("dispersion_index")),
                              _pct(c.get("share_top1_days")), _pct(c.get("share_top3_days")),
                              _pct(c.get("share_top10_days")), c.get("max_day", "-"),
                              c.get("max_day_count", 0)])
        if len(assets) >= 2:
            a, b = assets[0], assets[1]
            ra = [r for r in results if r.asset == a and r.setting == name][0]
            x = CO.cross_asset_concordance(all_events[(a, name)], all_events[(b, name)],
                                           ra.first_ms, ra.last_ms, tol_sec=args.concord_tol_sec)
            y = CO.cross_asset_concordance(all_events[(b, name)], all_events[(a, name)],
                                           ra.first_ms, ra.last_ms, tol_sec=args.concord_tol_sec)
            xa_rows.append([name, f"{a}->{b}", x["n_a"], x.get("matched", 0),
                            _pct(x.get("matched_frac")), _pct(x.get("chance_frac"), 2),
                            _f(x.get("lift"))])
            xa_rows.append([name, f"{b}->{a}", y["n_a"], y.get("matched", 0),
                            _pct(y.get("matched_frac")), _pct(y.get("chance_frac"), 2),
                            _f(y.get("lift"))])
    print("\nTemporal concentration (Poisson null gives dispersion index 1.0):")
    print(fmt_table(conc_rows, ["asset", "setting", "events", "days_hit", "days",
                                "dispersion", "top1d", "top3d", "top10d",
                                "busiest_day", "n"]))
    print(f"\nCross-asset concordance (independent order flow, tol={args.concord_tol_sec}s):")
    print(fmt_table(xa_rows, ["setting", "direction", "n", "matched", "matched%",
                              "chance%", "lift"]))
    for a in assets:
        daily = CO.load_daily(a, db_path=args.db)
        r0 = [r for r in results if r.asset == a and r.setting == "base"][0]
        e = CO.extreme_day_overlap(all_events[(a, "base")], daily,
                                   top_n=args.top_days,
                                   first_ms=r0.first_ms, last_ms=r0.last_ms)
        ex_rows.append([a, e.get("n_days_in_window", 0), e.get("n_events", 0),
                        e.get("events_on_top_days", 0),
                        _pct(e.get("frac_on_top_days")), _pct(e.get("chance_frac")),
                        _f(e.get("lift"))])
    print(f"\nOverlap with the {args.top_days} largest absolute daily-return days "
          f"WITHIN THE EVENT WINDOW (base setting):")
    print(fmt_table(ex_rows, ["asset", "days_in_win", "events", "on_top_days",
                              "share", "chance", "lift"]))
    print("  CIRCULARITY: condition M already requires an abnormal price move, so\n"
          "  agreement here is partly mechanical. Informative mainly in the negative.")

    _save(pd.DataFrame(conc_rows, columns=["asset", "setting", "events", "days_hit",
                                           "days", "dispersion_index", "share_top1",
                                           "share_top3", "share_top10", "busiest_day",
                                           "busiest_count"]),
          out_dir, "t1_concentration")
    _save(pd.DataFrame(xa_rows, columns=["setting", "direction", "n", "matched",
                                         "matched_frac", "chance_frac", "lift"]),
          out_dir, "t1_cross_asset_concordance")
    for (a, name), ev in all_events.items():
        if name == "base" and not ev.empty:
            e2 = ev.copy()
            e2["start_utc"] = [ms_to_str(t) for t in e2["start_ms"]]
            _save(e2, out_dir, f"t1_events_{a}_base")
    return 0


# ================================================================ T2 ======

def cmd_t2(args) -> int:
    out_dir = Path(args.out_dir)
    print(banner("T2 -- F1 TOKEN UNLOCKS: do circulating_supply jumps mark them?"))

    df = UN.load_supply(args.db)
    cov = UN.coverage_report(df)
    print("\nT2.1  USABLE circulating_supply SERIES (this is the gate):")
    print(fmt_table(cov.values.tolist(), list(cov.columns)))
    _save(cov, out_dir, "t2_coverage")

    print(banner("T2.2  JUMP DETECTION -- THRESHOLD SENSITIVITY"))
    rows, keep = [], {}
    for name, p in UN.DEFAULT_SWEEP:
        j = UN.detect_jumps(df, p)
        keep[name] = j
        span = float(cov["span_days"].max()) if len(cov) else 0.0
        for a in sorted(df["asset"].unique()):
            aj = j[j["asset"] == a] if len(j) else j
            rows.append([name, p.label(), a, len(aj),
                         int(aj["is_cliff"].sum()) if len(aj) else 0,
                         f"{per_year(len(aj), span):.1f}" if span else "-"])
    print()
    print(fmt_table(rows, ["setting", "params", "asset", "jumps", "cliffs", "jumps/yr"]))
    _save(pd.DataFrame(rows, columns=["setting", "params", "asset", "n_jumps",
                                      "n_cliffs", "jumps_per_year"]),
          out_dir, "t2_jump_sensitivity")

    print(banner("T2.3  CLIFF vs DRIFT -- which pattern dominates?"))
    # Decomposed on the LOOSE set: the base and strict sets are EMPTY, and a
    # cliff share computed from an empty numerator would read as DRIFT for
    # arithmetic reasons rather than measured ones.
    dec = UN.growth_decomposition(df, keep["loose"])
    print("\n(decomposed on the LOOSE jump set -- base and strict detect nothing "
          "at all,\n so a share computed from them would be 0 by arithmetic, "
          "not by measurement)")
    print(fmt_table(dec.values.tolist(), list(dec.columns)))
    _save(dec, out_dir, "t2_growth_decomposition")
    _save(keep["base"], out_dir, "t2_jumps_base")
    if len(keep["loose"]):
        print("\nAll detected increases at the LOOSE setting (the full candidate set):")
        show = keep["loose"][["asset", "date", "gap_days", "jump_frac",
                              "background_mad", "cliff_score", "scorable",
                              "is_cliff"]].copy()
        show["background_mad"] = show["background_mad"].map(lambda v: _f(v, 8))
        show["jump_frac"] = show["jump_frac"].map(lambda v: f"{100*v:.4f}%")
        show["cliff_score"] = show["cliff_score"].map(lambda v: _f(v))
        print(fmt_table(show.values.tolist(), list(show.columns)))
        _save(keep["loose"], out_dir, "t2_jumps_loose")

    v = UN.verdict(cov, keep, dec, pattern_setting="loose")
    print(banner("T2.4  VERDICT"))
    for k, val in v.items():
        print(f"  {k:<28} {val}")
    _save(v, out_dir, "t2_verdict")
    print("""
DATING PRECISION: market_data is a daily snapshot. An unlock can be dated no
more precisely than the observed sampling gap reported in T2.1 -- see
median_gap_days / max_gap_days there, not the nominal cadence.

SPOT-CHECK AGAINST PUBLIC UNLOCK RECORDS: deliberately NOT performed here. The
asset list this table actually covers is reported above; whether a spot-check is
even meaningful depends on it, and that is stated in the retro rather than
worked around with a proxy.""")
    return 0


# ================================================================ T3 ======

def cmd_t3(args) -> int:
    out_dir = Path(args.out_dir)
    print(banner("T3 -- D1 LEVERAGED TOKENS: what is the candidate universe?"))

    from engines.xsec_reversal.archive import ArchiveClient
    from engines.xsec_reversal.universe import DEFAULT_LEVERAGED_PATTERNS

    sym_file = Path(args.symbols_file) if args.symbols_file else None
    if sym_file and sym_file.exists():
        symbols = [s.strip() for s in sym_file.read_text(encoding="utf-8").split() if s.strip()]
        print(f"\nsymbol source: {sym_file} ({len(symbols)} symbols)")
    else:
        print("\nsymbol source: S3 bucket listing (survivorship-bias-free path, "
              "same as Cycle 60) -- requires network")
        client = ArchiveClient(cache_dir=Path(args.cache_dir))
        symbols = client.list_all_symbols(quote=None)
        print(f"  enumerated {len(symbols)} archive symbols INCLUDING DELISTED")
        if sym_file:
            ensure_dir(sym_file.parent)
            sym_file.write_text("\n".join(symbols) + "\n", encoding="utf-8")
            print(f"  cached -> {sym_file}")

    print("\nNOTE: data/external/xsec/symbols_all.txt is the POST-FILTER Cycle 60\n"
          "universe -- filter_symbol_names already removed every leveraged token,\n"
          "so reading it here would report a universe of ZERO.")

    cls = LV.classify(symbols)
    gen = cls[cls["is_genuine"]]
    print(banner("T3.1  CANDIDATE UNIVERSE"))
    print(f"\n{len(cls)} symbols PARSE as <BASE><UP|DOWN|BULL|BEAR><QUOTE>; "
          f"{len(gen)} are GENUINE\n(the implied underlying actually trades). "
          f"Genuine, by quote x suffix:")
    print(gen.groupby(["quote", "suffix"]).size().unstack(fill_value=0).to_string())
    u = gen[gen["quote"] == "USDT"].sort_values("symbol")
    print(f"\nUSDT-quoted genuine candidates ({len(u)}):")
    print(fmt_table(u[["symbol", "base", "suffix", "direction", "underlying",
                       "underlying_assumed"]].values.tolist(),
                    ["symbol", "base", "suffix", "dir", "underlying", "assumed"]))
    _save(cls, out_dir, "t3_candidates")

    fp = LV.substring_false_positives(symbols, DEFAULT_LEVERAGED_PATTERNS)
    print(banner("T3.2  SUBSTRING-RULE FALSE POSITIVES (a live Cycle 60 defect)"))
    print(f"\nDEFAULT_LEVERAGED_PATTERNS is matched with `in`, not `endswith`. "
          f"{len(fp)} REAL spot\nsymbols match it and were therefore deleted from "
          f"the Cycle 60 universe:")
    for f in fp:
        print(f"  {f['symbol']:<14} {f['reason']}")
    if not fp:
        print("  (none)")
    _save({"patterns": list(DEFAULT_LEVERAGED_PATTERNS), "false_positives": fp},
          out_dir, "t3_substring_false_positives")

    print(banner("T3.3  SECOND LEG -- is the underlying available?"))
    print(f"\n  name-parses with the underlying present in the archive: "
          f"{len(gen)}/{len(cls)}")
    missing = cls.loc[~cls["underlying_exists"], "symbol"].tolist()
    if missing:
        print(f"  NO underlying (hence NOT leveraged tokens): {missing}")
    assumed = gen.loc[gen["underlying_assumed"], "symbol"].tolist()
    if assumed:
        print(f"  underlying ASSUMED (bare BULL/BEAR, mapped to BTC): {assumed}")
    print("  ==> every genuine candidate has its underlying in the same archive,\n"
          "      so the second leg is available for the whole universe.")

    if args.coverage:
        print(banner("T3.4  DATE COVERAGE AND DELISTING (archive period listing)"))
        client = ArchiveClient(cache_dir=Path(args.cache_dir))
        syms = u["symbol"].tolist()   # genuine USDT candidates only
        if args.limit:
            syms = syms[: args.limit]
            print(f"  (limited to first {args.limit} symbols by --limit)")

        # The archive frontier must come from a symbol KNOWN to still trade.
        # Taking max(last_period) over the leveraged tokens themselves would be
        # circular -- if every one of them is delisted, the frontier would move
        # back to match and nothing would ever look delisted.
        frontier = args.archive_last_period
        if not frontier:
            ref = LV.periods_to_span(
                client.list_symbol_periods(args.frontier_symbol, args.interval))
            frontier = ref["last"]
            print(f"  archive frontier calibrated from {args.frontier_symbol} "
                  f"(known live): last period = {frontier}")

        t0 = time.time()
        cov = LV.coverage(client, syms, interval=args.interval,
                          archive_last_period=frontier)
        print(f"\n  {len(cov)} symbols listed in {time.time()-t0:.0f}s")
        print(fmt_table(cov.values.tolist(), list(cov.columns)))
        n_del = int(cov["delisted"].fillna(False).sum())
        tot_months = int(cov["n_months"].sum())
        last_alive = cov["last"].dropna().max() if len(cov) else None
        print(f"\n  delisted: {n_del}/{len(cov)}   "
              f"total symbol-months of {args.interval} data: {tot_months:,}")
        print(f"  scheduled daily resets implied: about {tot_months*30:,} "
              f"token-days\n  (D1 is the ONLY P1 scenario that is not event-rare)")
        if n_del == len(cov) and len(cov):
            print(f"\n  *** THE WHOLE PRODUCT CLASS IS DEAD. The most recent data for "
                  f"ANY\n      genuine leveraged token is {last_alive}, against an "
                  f"archive frontier of\n      {frontier}. D1 is therefore a "
                  f"historical-only study of a product that\n      can no longer be "
                  f"traded on this venue. State this before designing it.")
        _save(cov, out_dir, "t3_coverage")
    else:
        print("\n(T3.4 date coverage skipped -- pass --coverage; it makes one "
              "archive listing request per symbol)")
    return 0


# ================================================================ T4 ======

def cmd_t4(args) -> int:
    out_dir = Path(args.out_dir)
    print(banner("T4 -- OPEN INTEREST: how bad is the gap?"))

    scan = OI.scan_schema_for_oi(args.db)
    print(f"\nT4.1  IS OI ABSENT?  scanned {scan['n_tables']} tables")
    print(f"  tables: {', '.join(scan['tables'])}")
    print(f"  OI-shaped table names:  {scan['oi_table_hits'] or 'NONE'}")
    print(f"  OI-shaped column names: {scan['oi_column_hits'] or 'NONE'}")
    print(f"  ==> OI present in crypto_data.db: {scan['oi_present']}")
    _save({k: v for k, v in scan.items() if k != "columns_by_table"}, out_dir, "t4_schema_scan")

    print(banner("T4.2  WHAT compute_funding_regime ACTUALLY DOES ABOUT IT"))
    deg = OI.degradation_probe()
    print(f"\n  declared states for class F : {deg['declared_states']}")
    print(f"  reachable WITH OI           : {deg['reachable_with_oi']}")
    print(f"  reachable WITHOUT OI        : {deg['reachable_without_oi']}")
    print(f"  LOST states                 : {deg['lost_states']}")
    print(f"  raises when OI missing?     : {deg['errors_when_oi_missing']}")
    print(f"  ==> SILENTLY DEGRADES       : {deg['silently_degrades']}")
    if deg["silently_degrades"]:
        print("\n  MECHANISM: oi_change_7d is initialised to 0.0 and only overwritten\n"
              "  when oi_values is not None. States +/-2 require abs(oi_change_7d)>0.10,\n"
              "  so they are UNREACHABLE. Class F collapses from a 5-state axis to a\n"
              "  3-state one. Nothing raises, nothing logs, and RegimeState.missing\n"
              "  does NOT list F -- so the degradation is invisible downstream.")
    _save(deg, out_dir, "t4_degradation")

    ca = OI.caller_audit()
    print("\n  production callsites of RegimeEngine.compute:")
    for f in ca["callsites"]:
        print(f"    {f.get('file')}:{f.get('line')}  passes_oi={f.get('passes_oi')}")
    print(f"  ==> any caller supplies OI: {ca['any_passes_oi']}")
    _save(ca, out_dir, "t4_caller_audit")

    if args.network:
        print(banner("T4.3  IS OI COLLECTABLE FROM VENUES WE ALREADY POLL?"))
        probe = OI.probe_venue_oi(venues=tuple(args.venues), symbol=args.oi_symbol)
        print(f"\n  ccxt {probe.get('ccxt_version')}")
        for vid, v in probe.get("venues", {}).items():
            print(f"\n  --- {vid} ---")
            print(f"    has: {v.get('has')}")
            print(f"    live reading: {v.get('spot_reading')}")
            print(f"    active linear swaps pollable: {v.get('n_active_linear_swaps')}")
            for tf, d in (v.get("timeframes") or {}).items():
                print(f"    tf={tf:<4} {d}")
            print("    retention walk (1d bars):")
            for k, d in (v.get("retention_1d") or {}).items():
                print(f"      {k:<14} {d}")
        _save(probe, out_dir, "t4_venue_probe")
    else:
        print("\n(T4.3 venue probe skipped -- pass --network)")
    return 0


# ================================================================ T5 ======

def cmd_t5(args) -> int:
    out_dir = Path(args.out_dir)
    print(banner("T5 -- SCENARIO x REGIME GRID FEASIBILITY"))

    from engines.regime_engine import REGIME_CLASSES, REGIME_STATE_RANGES
    full = int(np.prod([len(REGIME_STATE_RANGES[c]) for c in REGIME_CLASSES]))
    marg = int(sum(len(REGIME_STATE_RANGES[c]) for c in REGIME_CLASSES))
    print(f"""
GRID ARITHMETIC FIRST -- the number that decides everything:
  full joint product of all 12 classes : {full:,} cells
  marginal (class, state) pairs        : {marg} cells per scenario
A four-month window on two assets cannot populate {full:,} cells, so occupancy is
reported MARGINALLY (per axis) and for one coarse 2-axis collapse. Reporting a
"12-class grid" as 12 cells would be the pooling the taxonomy warns about.""")

    p = OC.OccupancyParams(trailing_days=args.trailing_days,
                           step_hours=args.step_hours,
                           min_events_estimable=args.min_estimable)

    # --- events come from the T1 detector at the base setting -------------
    assets = args.assets
    hourly = {a: OC.load_hourly(a, db_path=args.db) for a in assets}
    universe = {a: h for a, h in hourly.items() if not h.empty}

    all_marg, all_joint, all_collapsed, summary = [], [], [], []
    for a in assets:
        if hourly[a].empty:
            print(f"\n[{a}] no ohlcv_1m -- regime cannot be computed. SKIPPED.")
            continue
        buckets = C.build_bucket_cache(a, bucket_sec=args.bucket_sec,
                                       db_path=args.db, cache_dir=args.cache_dir)
        # Same detector settings as the T1 base row, by reference not by copy.
        r = C.run_detector(a, buckets, args.bucket_sec, "base",
                           C.sweep_params("base"))
        fund = OC.load_funding(a, venue=args.funding_venue, db_path=args.db)
        t0 = time.time()
        reg = OC.regime_series(a, hourly[a], funding=fund, universe=universe,
                               p=p, first_ms=r.first_ms, last_ms=r.last_ms)
        print(f"\n[{a}] regime series: {len(reg)} evaluations in {time.time()-t0:.0f}s "
              f"(trailing {p.trailing_days}d, step {p.step_hours}h)")
        if len(reg):
            miss = reg["missing"].value_counts().head(5).to_dict()
            print(f"  classes the engine could NOT compute: {miss}")
        _save(reg, out_dir, f"t5_regime_series_{a}")

        deg = OC.axis_degeneracy(reg)
        eff = OC.effective_grid_size(deg)
        print(f"\n[{a}] AXIS DEGENERACY -- states each class ACTUALLY takes:")
        print(fmt_table(deg.values.tolist(), list(deg.columns)))
        print(f"\n[{a}] nominal joint grid {eff['nominal_joint_cells']:,} cells -> "
              f"effective {eff['effective_joint_cells']:,}")
        print(f"[{a}] nominal marginal {eff['nominal_marginal_cells']} cells -> "
              f"effective {eff['effective_marginal_cells']}")
        print(f"[{a}] degenerate (single-valued) axes: {eff['degenerate_axes']}  "
              f"-> {eff['informative_axes']}/12 axes carry information")
        _save(deg, out_dir, f"t5_axis_degeneracy_{a}")
        _save(eff, out_dir, f"t5_effective_grid_{a}")

        assigned = OC.assign_regime(r.events, reg)
        _save(assigned, out_dir, f"t5_events_with_regime_{a}")

        m = OC.marginal_occupancy(assigned, scenario=f"A2_cascades_{a}")
        all_marg.append(m)
        j = OC.joint_occupancy(assigned, class_x="B", class_y="G",
                               scenario=f"A2_cascades_{a}")
        all_joint.append(j)

        print(f"\n[{a}] MARGINAL occupancy -- events per (class, state), "
              f"n_events={len(assigned)}:")
        piv = m.pivot_table(index=["class", "class_name"], columns="state",
                            values="n_events", fill_value=0)
        print(piv.to_string())

        bo = OC.bucket_occupancy(m)
        print(f"\n[{a}] CELL-OCCUPANCY DISTRIBUTION (marginal, {len(m)} cells):")
        print(fmt_table(bo.values.tolist(), list(bo.columns)))
        v = OC.granularity_verdict(m, p)
        summary.append([a, "marginal(12 axes)", v["cells"], v["cells_empty"],
                        v["cells_estimable"], f"{v['pct_estimable']}%", v["buildable"]])

        bo_j = OC.bucket_occupancy(j)
        print(f"\n[{a}] JOINT-2 occupancy B(vol level) x G(liquidity), {len(j)} cells:")
        pj = j.pivot_table(index="x_state", columns="y_state", values="n_events",
                           fill_value=0)
        print(pj.to_string())
        print(fmt_table(bo_j.values.tolist(), list(bo_j.columns)))
        vj = OC.granularity_verdict(j, p)
        summary.append([a, "joint B x G (4x4)", vj["cells"], vj["cells_empty"],
                        vj["cells_estimable"], f"{vj['pct_estimable']}%", vj["buildable"]])

        k = OC.collapsed_joint_occupancy(assigned, class_x="B", class_y="G",
                                         scenario=f"A2_cascades_{a}")
        all_collapsed.append(k)
        print(f"\n[{a}] COLLAPSED 3x3 -- vol level x liquidity "
              f"(extreme states merged; see DEFAULT_COLLAPSE_3):")
        print(k.pivot_table(index="x_level", columns="y_level", values="n_events",
                            fill_value=0).to_string())
        bo_k = OC.bucket_occupancy(k)
        print(fmt_table(bo_k.values.tolist(), list(bo_k.columns)))
        vk = OC.granularity_verdict(k, p)
        summary.append([a, "collapsed B x G (3x3)", vk["cells"], vk["cells_empty"],
                        vk["cells_estimable"], f"{vk['pct_estimable']}%", vk["buildable"]])

    if all_marg:
        _save(pd.concat(all_marg, ignore_index=True), out_dir, "t5_marginal_occupancy")
    if all_joint:
        _save(pd.concat(all_joint, ignore_index=True), out_dir, "t5_joint_occupancy")
    if all_collapsed:
        _save(pd.concat(all_collapsed, ignore_index=True), out_dir,
              "t5_collapsed_occupancy")

    print(banner("T5.X  BUILDABLE / NOT-BUILDABLE"))
    print()
    print(fmt_table(summary, ["asset", "granularity", "cells", "empty",
                              f"estimable(>={p.min_events_estimable})", "pct", "buildable"]))
    _save(pd.DataFrame(summary, columns=["asset", "granularity", "cells", "empty",
                                         "estimable", "pct_estimable", "buildable"]),
          out_dir, "t5_verdict")
    print("""
SCENARIOS OTHER THAN A2 ARE NOT IN THIS TABLE, and that is a finding rather than
an omission:
  F1 unlocks -- regime needs intraday OHLCV; crypto_data.db holds ohlcv_1m for
    BTC and ETH only, while market_data covers ADA/BTC/ETH/SOL/XRP. The three
    non-BTC/ETH assets cannot be regime-assigned from this database at all.
  D1 leveraged tokens -- no event series exists yet (T3 is a universe audit, not
    a collection run), and the underlying klines are in the Binance archive
    rather than crypto_data.db. Its event count is bounded by construction:
    one scheduled reset per token per day.""")
    return 0


# =============================================================== driver ===

def cmd_all(args) -> int:
    rc = 0
    for fn in (cmd_t1, cmd_t2, cmd_t3, cmd_t4, cmd_t5):
        try:
            rc |= fn(args)
        except Exception as e:  # noqa: BLE001
            logger.exception("%s FAILED: %s", fn.__name__, e)
            print(f"\n*** {fn.__name__} FAILED: {type(e).__name__}: {e}")
            print("*** BLOCKER reported rather than worked around (brief constraint).")
            rc |= 1
    return rc


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cycle 61 forced-trade data audit (measurement only)")
    p.add_argument("-v", "--verbose", action="count", default=2,
                   help="default is -vv (Rule 25: maximum verbosity)")
    p.add_argument("-q", "--quiet", action="store_true")
    p.add_argument("--validate", dest="validate", action="store_true", default=True,
                   help="independent re-derivation checks (DEFAULT ON, Rule 25)")
    p.add_argument("--no-validate", dest="validate", action="store_false")
    p.add_argument("--validate-sample", type=int, default=25,
                   help="bucket-cache rows re-derived from source")
    p.add_argument("--validate-full-scan", action="store_true",
                   help="check side<->is_buyer_maker on EVERY row (one full "
                        "table scan per asset, minutes) instead of sampled windows")
    p.add_argument("--validate-sample-days", type=int, default=14,
                   help="total days sampled for the side-convention check")
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--seed", type=int, default=61)

    # T1
    p.add_argument("--bucket-sec", type=int, default=10,
                   help="base aggregation bucket for the trades cache")
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--window-sweep", type=int, nargs="+", default=[30, 60, 300, 900])
    p.add_argument("--baseline-hours", type=float, default=24.0)
    p.add_argument("--control-draws", type=int, default=500)
    p.add_argument("--concord-tol-sec", type=int, default=300)
    p.add_argument("--top-days", type=int, default=10)

    # T3
    p.add_argument("--symbols-file", default="claude/scratch/c61_archive_symbols_raw.txt")
    p.add_argument("--coverage", action="store_true",
                   help="list archive periods per symbol (one request each)")
    p.add_argument("--interval", default="1d")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--frontier-symbol", default="BTCUSDT",
                   help="known-live symbol used to calibrate the archive frontier")
    p.add_argument("--archive-last-period", default=None,
                   help="YYYY-MM frontier used for the delisted inference")

    # T4
    p.add_argument("--network", action="store_true", help="run the live CCXT OI probe")
    p.add_argument("--venues", nargs="+", default=["binance", "bybit"])
    p.add_argument("--oi-symbol", default="BTC/USDT:USDT")

    # T5
    p.add_argument("--assets", nargs="+", default=["BTC", "ETH"])
    p.add_argument("--trailing-days", type=int, default=90)
    p.add_argument("--step-hours", type=int, default=24)
    p.add_argument("--min-estimable", type=int, default=10)
    p.add_argument("--funding-venue", default="binance")

    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn, helptext in (
        ("t1-cascades", cmd_t1, "A2 liquidation cascades: counts + detectability"),
        ("t2-unlocks", cmd_t2, "F1 token unlocks: supply-jump detection"),
        ("t3-leveraged", cmd_t3, "D1 leveraged tokens: candidate universe"),
        ("t4-oi", cmd_t4, "open-interest gap assessment"),
        ("t5-occupancy", cmd_t5, "scenario x regime cell occupancy"),
        ("all", cmd_all, "run T1-T5 in order"),
    ):
        s = sub.add_parser(name, help=helptext)
        s.set_defaults(func=fn)
    return p


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    setup_logging(0 if args.quiet else args.verbose)
    ensure_dir(Path(args.out_dir))
    print(banner("CYCLE 61 -- FORCED-TRADE DATA AUDIT (measurement only)"))
    print(f"db         : {args.db}  (opened READ-ONLY)")
    print(f"out-dir    : {args.out_dir}")
    print(f"validate   : {args.validate}   verbose: {args.verbose}")
    print("NO strategy, NO P&L, NO Sharpe, NO backtest -- counts and feasibility only.")
    t0 = time.time()
    rc = args.func(args)
    print(f"\n[done in {time.time()-t0:.0f}s]")
    return rc


if __name__ == "__main__":
    sys.exit(main())
