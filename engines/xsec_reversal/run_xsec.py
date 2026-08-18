#!/usr/bin/env python3
"""
engines/xsec_reversal/run_xsec.py

CLI for the tiered cross-sectional reversal experiment (Cycle 60).

MODES
  symbols   Enumerate ALL archive symbols (incl. delisted) and write the list.
  collect   Download archive klines for the universe and build wide panels.
  backtest  Run the pre-registered tiered backtest grid + deflated Sharpe.

TYPICAL SEQUENCE (PowerShell, from repo root, venv active)
  python -m engines.xsec_reversal.run_xsec symbols -vv `
      --out data/external/xsec/symbols_all.txt

  python -m engines.xsec_reversal.run_xsec collect -vv `
      --symbols-file data/external/xsec/symbols_all.txt `
      --interval 4h --start 2021-01 --end 2026-06 `
      --out-dir data/external/xsec

  python -m engines.xsec_reversal.run_xsec backtest -vv `
      --panel-dir data/external/xsec --interval 4h `
      --out-dir outputs/xsec_reversal

EVERYTHING IS A PARAMETER. The pre-registered grid lives in
claude/handoffs/CYCLE60_XSEC_REVERSAL_PREREG.md; --grid runs it verbatim.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from engines.xsec_reversal.archive import (
    ArchiveClient, is_valid_symbol, month_range,
)
from engines.xsec_reversal.universe import (
    TierSpec, UniverseSpec, assign_tiers, build_point_in_time_universe,
    filter_symbol_names,
)
from engines.xsec_reversal.costs import (
    DEFAULT_TIER_SPREADS_BPS, CostSpec, estimate_spread_bps,
    tiered_spread_panel,
)
from engines.xsec_reversal.backtest import (
    BacktestSpec, SignalSpec, capacity_analysis, run_backtest,
)

logger = logging.getLogger("xsec.run")

PANEL_FIELDS = ("close", "high", "low", "dollar_vol")


def _setup_logging(v: int) -> None:
    level = logging.WARNING if v == 0 else (logging.INFO if v == 1 else logging.DEBUG)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


# --------------------------------------------------------------------------- #
def cmd_symbols(args) -> int:
    client = ArchiveClient(cache_dir=Path(args.cache_dir),
                           listing_url=args.listing_url)
    syms = client.list_all_symbols(quote=args.quote)
    spec = UniverseSpec(quote=args.quote,
                        exclude_leveraged=not args.keep_leveraged,
                        exclude_stables=not args.keep_stables)
    kept = filter_symbol_names(syms, spec)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(kept) + "\n", encoding="utf-8")
    print(f"enumerated {len(syms)} archive symbols (INCLUDING DELISTED)")
    if getattr(client, "rejected_symbols", None):
        print(f"rejected {len(client.rejected_symbols)} non-ticker bucket "
              f"prefix(es): {client.rejected_symbols[:10]}")
    print(f"after name filters: {len(kept)}")
    print(f"wrote {out}")
    print("NOTE: this list comes from the S3 bucket listing, NOT exchangeInfo -- "
          "that is the survivorship-bias fix. Do not regenerate it from the API.")
    return 0


def cmd_collect(args) -> int:
    # encoding is EXPLICIT: the file is written UTF-8, and Windows would
    # otherwise decode it as cp1252 and crash on any non-ASCII byte.
    raw = Path(args.symbols_file).read_text(encoding='utf-8')
    symbols = [s.strip() for s in raw.split() if s.strip()]
    bad = [s for s in symbols if not is_valid_symbol(s)]
    if bad:
        print(f'WARNING: dropping {len(bad)} invalid symbol name(s) from the list: {bad[:10]}')
        symbols = [s for s in symbols if is_valid_symbol(s)]
    if args.limit:
        symbols = symbols[: args.limit]
    periods = month_range(args.start, args.end)
    client = ArchiveClient(cache_dir=Path(args.cache_dir),
                           verify_checksum=not args.no_checksum)

    close, high, low, dvol = {}, {}, {}, {}
    n_ok = n_empty = n_err = 0
    t0 = time.time()
    total = len(symbols)
    print(f"collecting {total} symbols x {len(periods)} periods "
          f"({args.interval}) with {args.workers} worker(s) ...")

    def _fetch(sym):
        return sym, client.load_symbol_range(
            sym, args.interval, periods, use_listing=not args.no_listing)

    def _record(sym, df):
        nonlocal n_ok, n_empty
        if df is None or df.empty:
            n_empty += 1
            return
        close[sym] = df["close"]
        high[sym] = df["high"]
        low[sym] = df["low"]
        dvol[sym] = df["quote_asset_volume"]
        n_ok += 1

    def _progress(i):
        if i % 25 and i != total:
            return
        el = time.time() - t0
        rate = i / el if el > 0 else 0.0
        eta = (total - i) / rate if rate > 0 else float("nan")
        print(f"  ... {i}/{total} symbols ({n_ok} with data, {n_empty} empty, "
              f"{n_err} errors) | {el/60:.1f} min elapsed, ETA {eta/60:.1f} min")

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_fetch, s): s for s in symbols}
            for i, fut in enumerate(as_completed(futs), 1):
                sym = futs[fut]
                try:
                    sym, df = fut.result()
                    _record(sym, df)
                except Exception as e:  # noqa: BLE001
                    n_err += 1
                    logger.error("[%s] failed: %s", sym, e)
                _progress(i)
    else:
        for i, sym in enumerate(symbols, 1):
            try:
                _, df = _fetch(sym)
                _record(sym, df)
            except Exception as e:  # noqa: BLE001
                n_err += 1
                logger.error("[%s] failed: %s", sym, e)
            _progress(i)

    print(f"\ncollect finished in {(time.time()-t0)/60:.1f} min "
          f"({n_ok} ok, {n_empty} empty, {n_err} errors)")

    if not close:
        print("ERROR: no data collected")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panels = {
        "close": pd.DataFrame(close).sort_index(),
        "high": pd.DataFrame(high).sort_index(),
        "low": pd.DataFrame(low).sort_index(),
        "dollar_vol": pd.DataFrame(dvol).sort_index(),
    }
    for name, p in panels.items():
        fp = out_dir / f"panel_{name}_{args.interval}.parquet"
        p.to_parquet(fp)
        print(f"wrote {fp}  shape={p.shape}")

    cov = panels["close"].notna().sum()
    print(f"\nsymbols with data: {n_ok}; no data in window: {n_empty}; errors: {n_err}")
    print(f"bars per symbol: min={int(cov.min())} med={int(cov.median())} max={int(cov.max())}")
    print(f"panel span: {panels['close'].index.min()} -> {panels['close'].index.max()}")
    delisted = int((panels['close'].iloc[-1].isna() & (cov > 0)).sum())
    print(f"symbols with NO data in the final bar (delisted/inactive): {delisted}")
    print("  ^ a nonzero count here is EVIDENCE the survivorship fix is working.")
    return 0


def _load_panels(panel_dir: Path, interval: str) -> dict:
    out = {}
    for name in PANEL_FIELDS:
        fp = panel_dir / f"panel_{name}_{interval}.parquet"
        if not fp.exists():
            raise FileNotFoundError(f"missing panel: {fp} (run `collect` first)")
        out[name] = pd.read_parquet(fp)
    return out


def cmd_backtest(args) -> int:
    panels = _load_panels(Path(args.panel_dir), args.interval)
    close, high, low, dvol = (panels["close"], panels["high"],
                              panels["low"], panels["dollar_vol"])
    print(f"panels loaded: close={close.shape}, span {close.index.min()} -> {close.index.max()}")

    # *** THE SPLIT MUST ACTUALLY SLICE THE DATA. ***
    # The first version computed is_close/oos_close, PRINTED them, and then ran
    # the backtest on the full panel anyway -- so an "IS-only" grid silently
    # consumed the sealed OOS window. See CYCLE60_COST_MODEL_FINDING.md.
    if args.train_end:
        cut = pd.Timestamp(args.train_end, tz="UTC")
        if args.oos:
            close = close.loc[cut:]
            high = high.loc[cut:]
            low = low.loc[cut:]
            dvol = dvol.loc[cut:]
            print(f"*** OOS SLICE ACTIVE: {close.index.min()} -> {close.index.max()} "
                  f"({len(close)} bars) ***")
        else:
            close = close.loc[:cut]
            high = high.loc[:cut]
            low = low.loc[:cut]
            dvol = dvol.loc[:cut]
            print(f"*** IS SLICE ACTIVE: {close.index.min()} -> {close.index.max()} "
                  f"({len(close)} bars) ***")
    else:
        print("*** NO SPLIT: running on the FULL panel ***")

    cost_spec = CostSpec(
        spread_model=args.spread_model,
        fee_bps_per_side=args.fee_bps,
        extra_slippage_bps_per_side=args.extra_slippage_bps,
        spread_window_bars=args.spread_window,
        fixed_spread_bps=args.fixed_spread_bps,
    )
    if cost_spec.spread_model in ("corwin_schultz", "abdi_ranaldo",
                                  "max_of_estimators"):
        print("\n*** WARNING: OHLC spread estimators are NOT VALID on 4h crypto "
              "bars ***")
        print("    Anchor test: Corwin-Schultz reported 37-76 bps for "
              "BTC/ETH/BNB/XRP/SOL")
        print("    (true ~1-2 bps), ranked by VOLATILITY not spread, and "
              "inverted across")
        print("    tiers. Abdi-Ranaldo was degenerate. Use --spread-model "
              "tiered_fixed.")
        print("    Any net-of-cost number below is NOT a verdict.\n")
        spread_bps = estimate_spread_bps(high, low, close, cost_spec)
    else:
        spread_bps = None  # built after tiers are assigned

    uspec = UniverseSpec(
        adv_lookback_bars=args.adv_lookback,
        min_adv_usd=args.min_adv_usd,
        min_obs_in_window=args.min_obs_in_window,
        min_total_history_bars=args.min_history,
        max_symbols=args.max_symbols,
    )
    tspec = TierSpec(n_tiers=args.n_tiers)

    # ---- grid definition must come FIRST: the universe depends on the
    # rebalance frequency, which follows each config's HOLDING period.
    if args.grid:
        formations = [3, 6, 12, 24]
        holdings = [3, 6, 12, 24]
        quantiles = [0.1, 0.2, 0.3]
        resid = ["demean", "none"]
    else:
        formations = [args.formation]
        holdings = [args.holding]
        quantiles = [args.quantile]
        resid = [args.residualize]

    # *** NON-OVERLAPPING BY CONSTRUCTION. ***
    # The first version pinned rebalance_every to args.holding (default 6) for
    # every config in the grid. With h=24 that recomputed positions every 6 bars
    # while measuring 24-bar forward returns -- each return counted 4x, and the
    # Sharpe annualisation over-stated the independent-observation count by the
    # same factor (x2 in Sharpe at h=24). Rebalancing on the holding period
    # makes the return series non-overlapping.
    reb_freqs = sorted({args.rebalance_every or h for h in holdings})
    universes, reb_indices = {}, {}
    for rf in reb_freqs:
        idx = close.index[::rf]
        print(f"building point-in-time universe for rebalance_every={rf} "
              f"({len(idx)} rebalances) ...")
        u = assign_tiers(build_point_in_time_universe(close, dvol, idx, uspec), tspec)
        universes[rf], reb_indices[rf] = u, idx
    uni = universes[reb_freqs[0]]
    reb_every = reb_freqs[0]

    per_dt = uni[uni["eligible"]].groupby("dt").size()
    if per_dt.empty or per_dt.median() == 0:
        print("\nFATAL: the point-in-time universe is EMPTY at every rebalance.")
        print("  Check --min-obs-in-window (<= --adv-lookback), --min-history,")
        print("  and --min-adv-usd. Refusing to produce meaningless metrics.")
        return 1
    print(f"eligible names/rebalance: min={int(per_dt.min())} "
          f"med={int(per_dt.median())} max={int(per_dt.max())}")

    need = args.min_names * args.n_tiers
    if per_dt.median() < need:
        print(f"\nFATAL: median eligible names ({int(per_dt.median())}) is below "
              f"{need} = --min-names x --n-tiers.")
        print("  Tiers would be too thin to rank. Loosen the filters or reduce "
              "--n-tiers / --min-names. Refusing to run.")
        return 1

    combos = [c for c in itertools.product(formations, holdings, quantiles, resid)]

    tier_spreads = dict(DEFAULT_TIER_SPREADS_BPS)
    if args.tier_spreads:
        vals = [float(x) for x in args.tier_spreads.split(",")]
        labels = sorted(uni["tier"].dropna().unique())
        tier_spreads = dict(zip(labels, vals))

    multipliers = ([float(m) for m in args.sensitivity.split(",")]
                   if args.sensitivity else [1.0])

    print(f"\nrunning {len(combos)} config(s) x {args.n_tiers} tiers "
          f"x {len(multipliers)} spread multiplier(s) ...")
    if cost_spec.spread_model == "tiered_fixed":
        print(f"assumed tier spreads (bps): {tier_spreads}")
        if len(multipliers) > 1:
            print(f"sensitivity multipliers: {multipliers}")
    print()

    records, trial_sharpes = [], []
    spread_cache = {}
    for mult in multipliers:
      for (f, h, q, r) in combos:
        rf = args.rebalance_every or h          # non-overlapping
        uni_h = universes[rf]
        if cost_spec.spread_model == "tiered_fixed":
            key = (rf, mult)
            if key not in spread_cache:
                spread_cache[key] = tiered_spread_panel(
                    uni_h, close.index, close.columns, tier_spreads, multiplier=mult)
            spread_bps = spread_cache[key]
        sspec = SignalSpec(formation_bars=f, holding_bars=h, quantile=q,
                           residualize_mode=r,
                           execution_lag_bars=args.execution_lag,
                           min_symbols_per_tier=args.min_names)
        bspec = BacktestSpec(periods_per_year=args.periods_per_year,
                             rebalance_every_bars=rf,
                             apply_costs=True)
        res = run_backtest(close, uni_h, spread_bps, sspec, bspec, cost_spec)
        for tier, r_ in res.items():
            rec = {"formation": f, "holding": h, "quantile": q,
                   "residualize": r, "spread_mult": mult,
                   "rebalance_every": rf, "tier": tier, **r_.metrics}
            records.append(rec)
            if mult == 1.0 and not np.isnan(r_.metrics.get("net_sharpe", np.nan)):
                trial_sharpes.append(r_.metrics["net_sharpe"])

    df = pd.DataFrame(records)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "grid_results.csv", index=False)

    print("\n" + "=" * 78)
    print("PER-TIER SUMMARY (best net Sharpe per tier)")
    print("=" * 78)
    for tier, grp in df.groupby("tier"):
        best = grp.loc[grp["net_sharpe"].idxmax()] if grp["net_sharpe"].notna().any() else None
        if best is None:
            print(f"{tier}: no valid results")
            continue
        print(f"{tier}: gross_sharpe={best['gross_sharpe']:.3f} "
              f"net_sharpe={best['net_sharpe']:.3f} "
              f"gross={best['gross_mean_bps']:.2f}bps net={best['net_mean_bps']:.2f}bps "
              f"cost={best['avg_cost_bps']:.1f}bps IC={best['mean_ic']:+.4f} "
              f"(f={int(best['formation'])},h={int(best['holding'])},"
              f"q={best['quantile']},{best['residualize']})")

    if len(multipliers) > 1:
        print("\n" + "=" * 78)
        print("SPREAD SENSITIVITY -- best net Sharpe per tier at each assumption")
        print("=" * 78)
        piv = (df.groupby(["tier", "spread_mult"])["net_sharpe"].max()
                 .unstack("spread_mult"))
        hdr = "  ".join(f"{m:>7.2f}x" for m in piv.columns)
        print(f"{'tier':14s} {hdr}")
        for tier, row in piv.iterrows():
            cells = "  ".join(f"{v:>8.2f}" for v in row.values)
            print(f"{tier:14s} {cells}")
        print("\nA tier is only a candidate edge if it stays above the 0.85 "
              "noise floor")
        print("across a PLAUSIBLE range of spread assumptions -- not just the "
              "cheapest one.")

    # ---- deflated Sharpe, computed within COHERENT CELLS ------------------
    # Configs with different holding periods have different annualisation
    # factors and different n_obs, so pooling them into one "trial population"
    # is incoherent. Within a (tier, holding, spread_mult) cell all 24 configs
    # share reb_every and n_periods, so the DSR is well posed there.
    print("\n" + "=" * 78)
    print("DEFLATED SHARPE -- per (tier, holding) cell at spread multiplier 1.0")
    print("=" * 78)
    try:
        from engines.infobar_lstm.deflated_sharpe import deflated_sharpe_ratio

        base = df[df["spread_mult"] == 1.0]
        print(f"{'tier':14s} {'h':>3s} {'trials':>7s} {'n_obs':>7s} "
              f"{'best_SR_ann':>12s} {'DSR':>8s}  verdict")
        best_by_tier = {}
        for (tier, h), cell in base.groupby(["tier", "holding"]):
            sr = cell["net_sharpe"].dropna()
            if len(sr) < 3:
                continue
            ann = np.sqrt(args.periods_per_year / float(h))
            n_obs = int(cell["n_periods"].max())
            try:
                d = deflated_sharpe_ratio(
                    observed_sr=float(sr.max()) / ann,
                    var_sr_across_trials=float(sr.var(ddof=1)) / (ann ** 2),
                    n_trials=int(len(sr)), n_obs=n_obs, skew=0.0, kurt=3.0)
                dsr = float(d.dsr)
            except Exception:  # noqa: BLE001
                continue
            verdict = "PASS" if dsr > 0.95 else "fail"
            print(f"{tier:14s} {int(h):3d} {len(sr):7d} {n_obs:7d} "
                  f"{sr.max():12.3f} {dsr:8.4f}  {verdict}")
            if dsr > best_by_tier.get(tier, (-1, None))[0]:
                best_by_tier[tier] = (dsr, h)
        print("\nNOTE: selecting the best CELL adds further multiplicity on top of")
        print("the within-cell correction, so these DSRs are an UPPER bound on")
        print("significance, not a final verdict.")
    except Exception as e:  # noqa: BLE001
        logger.warning("DSR step skipped: %s", e)

    # ---- criterion 4: demean must beat none -------------------------------
    print("\n" + "=" * 78)
    print("CRITERION 4 -- residualisation (demean should beat none)")
    print("=" * 78)
    b1 = df[df["spread_mult"] == 1.0]
    for tier, g in b1.groupby("tier"):
        dm = g[g["residualize"] == "demean"]["net_sharpe"]
        nn = g[g["residualize"] == "none"]["net_sharpe"]
        if dm.empty or nn.empty:
            continue
        ok = "OK" if dm.max() > nn.max() else "VIOLATED (beta dispersion?)"
        print(f"{tier:14s} demean best={dm.max():7.3f}  none best={nn.max():7.3f}"
              f"   median demean={dm.median():7.3f} none={nn.median():7.3f}   {ok}")

    cap = capacity_analysis(uni, SignalSpec(quantile=args.quantile),
                            participation_rate=args.participation)
    cap.to_csv(out_dir / "capacity.csv", index=False)
    print("\n" + "=" * 78)
    print(f"CAPACITY (at {args.participation:.1%} ADV participation)")
    print("=" * 78)
    for _, r_ in cap.iterrows():
        print(f"{r_['tier']}: median_ADV=${r_['median_adv_usd']:,.0f} "
              f"names/leg={int(r_['names_per_leg'])} "
              f"max_book=${r_['max_book_usd']:,.0f}")

    print(f"\nwrote {out_dir/'grid_results.csv'} and {out_dir/'capacity.csv'}")
    print("\nREMINDER: a NEGATIVE result here stays PROVISIONAL until the Cycle 59 "
          "framework validation lands (an unvalidated instrument cannot "
          "distinguish 'no edge' from 'broken measurement').")
    return 0



def cmd_diagnose(args) -> int:
    """Validate the SPREAD ESTIMATOR before trusting any net-of-cost verdict.

    Binance klines carry no quotes, so spreads are ESTIMATED from OHLC. The
    decisive check is an anchor test: BTCUSDT's true effective spread is about
    1-2 bps. If the estimator says 40 bps for BTC, every tier's cost is
    inflated by the same order and the net result is meaningless.
    """
    panels = _load_panels(Path(args.panel_dir), args.interval)
    close, high, low, dvol = (panels["close"], panels["high"],
                              panels["low"], panels["dollar_vol"])
    print(f"panels: {close.shape}, {close.index.min()} -> {close.index.max()}\n")

    cs = estimate_spread_bps(high, low, close,
                             CostSpec(spread_model="corwin_schultz",
                                      spread_window_bars=args.spread_window,
                                      min_spread_bps=0.0, max_spread_bps=1e4))
    ar = estimate_spread_bps(high, low, close,
                             CostSpec(spread_model="abdi_ranaldo",
                                      spread_window_bars=args.spread_window,
                                      min_spread_bps=0.0, max_spread_bps=1e4))

    print("=" * 78)
    print("ANCHOR TEST -- known-liquid symbols (true effective spread ~1-5 bps)")
    print("=" * 78)
    print(f"{'symbol':14s} {'Corwin-Schultz':>16s} {'Abdi-Ranaldo':>16s}   verdict")
    anchors = [a.strip().upper() for a in args.anchors.split(",")]
    inflation = []
    for a in anchors:
        if a not in close.columns:
            print(f"{a:14s} {'(absent)':>16s}")
            continue
        c = float(np.nanmedian(cs[a].values)) if cs[a].notna().any() else float("nan")
        r = float(np.nanmedian(ar[a].values)) if ar[a].notna().any() else float("nan")
        worst = np.nanmax([c, r])
        verd = "PLAUSIBLE" if worst <= args.anchor_tolerance_bps else "INFLATED"
        if np.isfinite(worst):
            inflation.append(worst / 2.0)  # vs a ~2 bps truth
        print(f"{a:14s} {c:16.2f} {r:16.2f}   {verd}")

    if inflation:
        factor = float(np.median(inflation))
        print(f"\nmedian anchor estimate is ~{factor:.1f}x a 2 bps truth")
        if factor > args.anchor_tolerance_bps / 2.0:
            print("VERDICT: the OHLC spread estimator is INFLATED on 4h crypto bars.")
            print("  Corwin-Schultz assumes range = volatility + spread with vol")
            print("  scaling in time. On high-vol 4h bars volatility dominates, and")
            print("  flooring the frequent negative estimates at zero biases the")
            print("  average UP. 'max_of_estimators' then compounds it.")
            print("  -> do NOT read the net-of-cost result as a verdict.")
        else:
            print("VERDICT: estimator looks usable at these horizons.")

    print("\n" + "=" * 78)
    print("UNIVERSE-WIDE SPREAD DISTRIBUTION (bps)")
    print("=" * 78)
    for name, est in (("corwin_schultz", cs), ("abdi_ranaldo", ar)):
        v = est.values[~np.isnan(est.values)]
        if v.size:
            print(f"{name:16s} p25={np.percentile(v,25):7.2f} "
                  f"med={np.percentile(v,50):7.2f} p75={np.percentile(v,75):7.2f} "
                  f"p95={np.percentile(v,95):7.2f}")

    print("\n" + "=" * 78)
    print("BREAKEVEN ANALYSIS (what spread would each tier tolerate?)")
    print("=" * 78)
    print("Using the probe's gross alpha per rebalance and turnover ~1.6:\n")
    print(f"{'tier':14s} {'gross bps':>10s} {'fee/reb':>9s} {'max spread':>12s}")
    for tier, gross in [("T1_liquid", args.t1_gross), ("T2_mid", args.t2_gross),
                        ("T3_illiquid", args.t3_gross)]:
        fee_cost = args.turnover * args.fee_bps
        max_one_way = gross / args.turnover
        max_spread = 2.0 * (max_one_way - args.fee_bps)
        print(f"{tier:14s} {gross:10.2f} {fee_cost:9.2f} {max_spread:12.2f}")
    print(f"\nAt fee={args.fee_bps} bps/side and turnover={args.turnover}, fees alone")
    print(f"cost {args.turnover*args.fee_bps:.1f} bps per rebalance BEFORE any spread.")
    print("If that already exceeds a tier's gross alpha, no spread model can save it")
    print("at this rebalance frequency -- the lever is a LONGER HOLDING PERIOD")
    print("(the pre-registered grid includes h=12 and h=24) or maker-side execution.")
    return 0


def _add_verbose(parser: argparse.ArgumentParser) -> None:
    """Attach -v/--verbose to a SUBparser.

    default=SUPPRESS is deliberate: without it, the subparser writes its own
    default (0) into the shared namespace and silently clobbers a top-level
    `-vv`. With SUPPRESS the attribute is only set when the flag is actually
    given, so `-vv symbols` and `symbols -vv` both work.
    """
    parser.add_argument("-v", "--verbose", action="count",
                        default=argparse.SUPPRESS,
                        help="-v=INFO, -vv=DEBUG (accepted before or after the subcommand)")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tiered cross-sectional reversal (Cycle 60)")
    p.add_argument("-v", "--verbose", action="count", default=0,
                   help="-v=INFO, -vv=DEBUG (accepted before or after the subcommand)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("symbols", help="enumerate ALL archive symbols (incl. delisted)")
    _add_verbose(s)
    s.add_argument("--out", default="data/external/xsec/symbols_all.txt")
    s.add_argument("--cache-dir", default="data/external/binance_archive")
    s.add_argument("--quote", default="USDT")
    s.add_argument("--listing-url", default=None,
                   help="S3 origin for bucket enumeration; default tries "
                        "the known candidates. NOTE: the data.binance.vision "
                        "CDN returns an HTML browser page for listing queries.")
    s.add_argument("--keep-leveraged", action="store_true")
    s.add_argument("--keep-stables", action="store_true")
    s.set_defaults(func=cmd_symbols)

    c = sub.add_parser("collect", help="download klines and build panels")
    _add_verbose(c)
    c.add_argument("--symbols-file", required=True)
    c.add_argument("--interval", default="4h")
    c.add_argument("--start", required=True, help="YYYY-MM")
    c.add_argument("--end", required=True, help="YYYY-MM")
    c.add_argument("--out-dir", default="data/external/xsec")
    c.add_argument("--cache-dir", default="data/external/binance_archive")
    c.add_argument("--limit", type=int, default=None)
    c.add_argument("--no-checksum", action="store_true",
                   help="skip .CHECKSUM verification (halves request count)")
    c.add_argument("--workers", type=int, default=6,
                   help="parallel symbol downloads (default 6; be polite)")
    c.add_argument("--no-listing", action="store_true",
                   help="probe every month instead of listing available "
                        "periods first (much slower; debug only)")
    c.set_defaults(func=cmd_collect)

    b = sub.add_parser("backtest", help="run the tiered backtest")
    _add_verbose(b)
    b.add_argument("--panel-dir", default="data/external/xsec")
    b.add_argument("--interval", default="4h")
    b.add_argument("--out-dir", default="outputs/xsec_reversal")
    b.add_argument("--train-end", default=None,
                   help="split date; data IS ACTUALLY SLICED at this point")
    b.add_argument("--oos", action="store_true",
                   help="run on the OOS slice (after --train-end) instead of IS")
    b.add_argument("--grid", action="store_true", help="run the pre-registered grid")
    b.add_argument("--formation", type=int, default=6)
    b.add_argument("--holding", type=int, default=6)
    b.add_argument("--rebalance-every", type=int, default=None)
    b.add_argument("--quantile", type=float, default=0.2)
    b.add_argument("--residualize", default="demean", choices=["demean", "beta", "none"])
    b.add_argument("--execution-lag", type=int, default=1)
    b.add_argument("--n-tiers", type=int, default=3)
    b.add_argument("--min-names", type=int, default=10)
    b.add_argument("--adv-lookback", type=int, default=180)
    b.add_argument("--min-adv-usd", type=float, default=50_000.0)
    b.add_argument("--min-history", type=int, default=200,
                   help="min TOTAL observed bars since listing (maturity)")
    b.add_argument("--min-obs-in-window", type=int, default=144,
                   help="min non-missing bars WITHIN the ADV lookback "
                        "window; must be <= --adv-lookback")
    b.add_argument("--max-symbols", type=int, default=None)
    b.add_argument("--spread-model", default="tiered_fixed",
                   choices=["tiered_fixed", "fixed", "corwin_schultz",
                            "abdi_ranaldo", "max_of_estimators"])
    b.add_argument("--spread-window", type=int, default=180)
    b.add_argument("--fee-bps", type=float, default=10.0)
    b.add_argument("--fixed-spread-bps", type=float, default=10.0,
                   help="used when --spread-model fixed")
    b.add_argument("--extra-slippage-bps", type=float, default=0.0)
    b.add_argument("--periods-per-year", type=float, default=2190.0)
    b.add_argument("--participation", type=float, default=0.01)
    b.add_argument("--tier-spreads", default=None,
                   help="comma-separated assumed effective spreads in bps, "
                        "one per tier ascending by tier label, e.g. '3,15,40'")
    b.add_argument("--sensitivity", default=None,
                   help="comma-separated spread multipliers to sweep, "
                        "e.g. '0.5,1,2,4'")
    b.set_defaults(func=cmd_backtest)

    d = sub.add_parser("diagnose", help="validate the spread estimator + breakeven")
    _add_verbose(d)
    d.add_argument("--panel-dir", default="data/external/xsec")
    d.add_argument("--interval", default="4h")
    d.add_argument("--spread-window", type=int, default=180)
    d.add_argument("--anchors", default="BTCUSDT,ETHUSDT,BNBUSDT,XRPUSDT,SOLUSDT",
                   help="known-liquid symbols whose true spread is ~1-5 bps")
    d.add_argument("--anchor-tolerance-bps", type=float, default=10.0)
    d.add_argument("--fee-bps", type=float, default=10.0)
    d.add_argument("--turnover", type=float, default=1.6)
    d.add_argument("--t1-gross", type=float, default=8.51)
    d.add_argument("--t2-gross", type=float, default=17.28)
    d.add_argument("--t3-gross", type=float, default=22.03)
    d.set_defaults(func=cmd_diagnose)
    return p


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
