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
import sys
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
from engines.xsec_reversal.costs import CostSpec, estimate_spread_bps
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
    n_ok = n_empty = 0
    for i, sym in enumerate(symbols, 1):
        try:
            df = client.load_symbol_range(sym, args.interval, periods)
        except Exception as e:  # noqa: BLE001
            logger.error("[%s] failed: %s", sym, e)
            continue
        if df is None or df.empty:
            n_empty += 1
            continue
        close[sym] = df["close"]
        high[sym] = df["high"]
        low[sym] = df["low"]
        dvol[sym] = df["quote_asset_volume"]
        n_ok += 1
        if i % 25 == 0:
            print(f"  ... {i}/{len(symbols)} symbols processed ({n_ok} with data)")

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
    print(f"\nsymbols with data: {n_ok}; no data in window: {n_empty}")
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

    if args.train_end:
        cut = pd.Timestamp(args.train_end, tz="UTC")
        is_close = close.loc[:cut]
        oos_close = close.loc[cut:]
        print(f"IS: {is_close.index.min()} -> {is_close.index.max()} ({len(is_close)} bars)")
        print(f"OOS: {oos_close.index.min()} -> {oos_close.index.max()} ({len(oos_close)} bars)")

    cost_spec = CostSpec(
        spread_model=args.spread_model,
        fee_bps_per_side=args.fee_bps,
        extra_slippage_bps_per_side=args.extra_slippage_bps,
        spread_window_bars=args.spread_window,
    )
    print(f"\nestimating spreads [{cost_spec.spread_model}] ...")
    spread_bps = estimate_spread_bps(high, low, close, cost_spec)

    uspec = UniverseSpec(
        adv_lookback_bars=args.adv_lookback,
        min_adv_usd=args.min_adv_usd,
        min_history_bars=args.min_history,
        max_symbols=args.max_symbols,
    )
    tspec = TierSpec(n_tiers=args.n_tiers)

    reb_every = args.rebalance_every or args.holding
    reb_index = close.index[::reb_every]
    print(f"building point-in-time universe over {len(reb_index)} rebalances ...")
    uni = build_point_in_time_universe(close, dvol, reb_index, uspec)
    uni = assign_tiers(uni, tspec)

    per_dt = uni[uni["eligible"]].groupby("dt").size()
    print(f"eligible names/rebalance: min={int(per_dt.min())} "
          f"med={int(per_dt.median())} max={int(per_dt.max())}")

    # ---- grid ------------------------------------------------------------
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

    combos = [c for c in itertools.product(formations, holdings, quantiles, resid)]
    print(f"\nrunning {len(combos)} config(s) x {args.n_tiers} tiers ...\n")

    records, trial_sharpes = [], []
    for (f, h, q, r) in combos:
        sspec = SignalSpec(formation_bars=f, holding_bars=h, quantile=q,
                           residualize_mode=r,
                           execution_lag_bars=args.execution_lag,
                           min_symbols_per_tier=args.min_names)
        bspec = BacktestSpec(periods_per_year=args.periods_per_year,
                             rebalance_every_bars=reb_every,
                             apply_costs=True)
        res = run_backtest(close, uni, spread_bps, sspec, bspec, cost_spec)
        for tier, r_ in res.items():
            rec = {"formation": f, "holding": h, "quantile": q,
                   "residualize": r, "tier": tier, **r_.metrics}
            records.append(rec)
            if not np.isnan(r_.metrics.get("net_sharpe", np.nan)):
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

    # ---- deflated Sharpe on the grid ------------------------------------
    if len(trial_sharpes) > 1:
        try:
            from engines.infobar_lstm.deflated_sharpe import (
                deflated_sharpe_ratio, trial_sharpe_stats,
            )
            stats = trial_sharpe_stats(trial_sharpes)
            best_sr = max(trial_sharpes)
            n_periods = int(df["n_periods"].max())
            dsr = deflated_sharpe_ratio(
                observed_sr=best_sr / np.sqrt(args.periods_per_year / reb_every),
                var_sr_across_trials=stats["var"] / (args.periods_per_year / reb_every),
                n_trials=stats["n"], n_obs=n_periods, skew=0.0, kurtosis=3.0)
            print("\n" + "=" * 78)
            print("DEFLATED SHARPE (multiple-testing correction)")
            print("=" * 78)
            print(f"trials={stats['n']}  best_ann_sharpe={best_sr:.3f}")
            print(f"expected-max-Sharpe under null (per-period)={dsr.sr0_expected_max:.4f}")
            print(f"DSR={dsr.dsr:.4f}   "
                  f"{'PASSES' if dsr.dsr > 0.95 else 'FAILS'} the 0.95 pre-registered gate")
            print("NOTE: skew/kurtosis passed as normal defaults here; feed the "
                  "realised moments of the winning config for the final verdict.")
        except Exception as e:  # noqa: BLE001
            logger.warning("DSR step skipped: %s", e)

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
    c.add_argument("--no-checksum", action="store_true")
    c.set_defaults(func=cmd_collect)

    b = sub.add_parser("backtest", help="run the tiered backtest")
    _add_verbose(b)
    b.add_argument("--panel-dir", default="data/external/xsec")
    b.add_argument("--interval", default="4h")
    b.add_argument("--out-dir", default="outputs/xsec_reversal")
    b.add_argument("--train-end", default=None, help="IS/OOS split date")
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
    b.add_argument("--min-history", type=int, default=200)
    b.add_argument("--max-symbols", type=int, default=None)
    b.add_argument("--spread-model", default="max_of_estimators",
                   choices=["corwin_schultz", "abdi_ranaldo", "fixed", "max_of_estimators"])
    b.add_argument("--spread-window", type=int, default=180)
    b.add_argument("--fee-bps", type=float, default=10.0)
    b.add_argument("--extra-slippage-bps", type=float, default=0.0)
    b.add_argument("--periods-per-year", type=float, default=2190.0)
    b.add_argument("--participation", type=float, default=0.01)
    b.set_defaults(func=cmd_backtest)
    return p


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
