"""
scripts/cycle57_basis_sharpe.py
===============================
Cycle 57 P2+P3 — the REAL, basis-inclusive risk-adjusted carry.

P&L per hour for long-spot + short-perp (delta-neutral), as a fraction of notional:
    pnl_t = funding_t + (spot_ret_t - perp_ret_t)
          = funding_t  -  Δbasis_t          (to first order)
where the short RECEIVES funding when funding>0, and (spot_ret - perp_ret) is
the realized basis/tracking P&L between the spot venue and the perp venue.

Structures measured:
  HL   : perp=hyperliquid, spot=coinbase (realistic Canada-legal hedge); funding=HL hourly
         (+ a binance-spot cross-check leg)
  CEX  : perp=binance,     spot=binance  (same-venue anchor);            funding=binance 8h
The CEX same-venue anchor reproduces the regime that makes atlas Exp 13's +4.65
high (basis ~ 0 same-venue, funding-dominated). The HL cross-venue number is the
degradation that actually decides deployment.

Reports per (asset, structure, window): funding_ann, basis mean/vol, basis-drift,
P&L vol, GROSS Sharpe, and cost-adjusted Sharpe at 9/20/35 bps. Windows: full,
2024, 2025, last-90d (the crux). Read-only over the research + production DBs.
"""
from __future__ import annotations

import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESEARCH_DB = REPO / "data" / "defi_funding_research.db"
PROD_DB = REPO / "data" / "crypto_data.db"

ASSETS = ["BTC", "ETH", "SOL", "XRP", "ADA", "AVAX", "DOGE", "LINK", "UNI", "LTC"]
HOURS_PER_YEAR = 24 * 365
NOW = datetime(2026, 6, 22, tzinfo=timezone.utc)

WINDOWS = {
    "full":     (datetime(2024, 1, 1, tzinfo=timezone.utc), NOW),
    "2024":     (datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2025, 1, 1, tzinfo=timezone.utc)),
    "2025":     (datetime(2025, 1, 1, tzinfo=timezone.utc), datetime(2026, 1, 1, tzinfo=timezone.utc)),
    "last90d":  (NOW - timedelta(days=90), NOW),
}
COSTS_BPS = [9.0, 20.0, 35.0]
ROLL_DAYS = 7  # cost amortization: round-trip paid once per 7-day hold


def _prices(conn, asset, venue, market) -> pd.Series:
    df = pd.read_sql(
        "SELECT timestamp, close FROM prices WHERE asset=? AND venue=? AND market=? "
        "ORDER BY timestamp", conn, params=(asset, venue, market))
    if df.empty:
        return pd.Series(dtype=float)
    s = pd.Series(df["close"].values,
                  index=pd.to_datetime(df["timestamp"], unit="ms", utc=True))
    return s[~s.index.duplicated()].sort_index()


def _hl_funding(conn, asset) -> pd.Series:
    df = pd.read_sql(
        "SELECT timestamp, funding_rate FROM funding_rates WHERE asset=? AND "
        "venue='hyperliquid' ORDER BY timestamp", conn, params=(asset,))
    if df.empty:
        return pd.Series(dtype=float)
    s = pd.Series(df["funding_rate"].values,
                  index=pd.to_datetime(df["timestamp"], unit="ms", utc=True))
    return s[~s.index.duplicated()].sort_index()


def _binance_funding_hourly(conn, asset) -> pd.Series:
    """binance 8h funding -> spread to hourly (rate/8 over the 8h), so it aligns
    with hourly basis for an apples-to-apples hourly Sharpe."""
    df = pd.read_sql(
        "SELECT timestamp, funding_rate FROM funding_rates WHERE asset=? AND "
        "venue='binance' ORDER BY timestamp", conn, params=(asset,))
    if df.empty:
        return pd.Series(dtype=float)
    s = pd.Series(df["funding_rate"].values,
                  index=pd.to_datetime(df["timestamp"], unit="ms", utc=True))
    s = s[~s.index.duplicated()].sort_index()
    hourly_idx = pd.date_range(s.index[0], NOW, freq="1h", tz="UTC")
    return (s.reindex(hourly_idx, method="ffill") / 8.0).dropna()


def _build(perp: pd.Series, spot: pd.Series, funding: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"perp": perp, "spot": spot, "funding": funding}).dropna()
    if len(df) < 50:
        return df
    df["basis"] = (df["perp"] - df["spot"]) / df["spot"]
    df["spot_ret"] = df["spot"].pct_change()
    df["perp_ret"] = df["perp"].pct_change()
    df["dbasis"] = df["basis"].diff()
    df["pnl"] = df["funding"] + (df["spot_ret"] - df["perp_ret"])
    return df.dropna()


def _metrics(df: pd.DataFrame, lo, hi) -> dict | None:
    w = df[(df.index >= lo) & (df.index < hi)]
    if len(w) < 50:
        return None
    funding_ann = w["funding"].mean() * HOURS_PER_YEAR
    basis_drift_ann = (w["spot_ret"] - w["perp_ret"]).mean() * HOURS_PER_YEAR
    vol_ann = w["pnl"].std() * np.sqrt(HOURS_PER_YEAR)
    mean_ann = w["pnl"].mean() * HOURS_PER_YEAR
    basis_vol_ann = w["dbasis"].std() * np.sqrt(HOURS_PER_YEAR)
    sharpe_gross = mean_ann / vol_ann if vol_ann > 0 else float("nan")
    out = {
        "n": len(w), "funding_ann": funding_ann, "basis_drift_ann": basis_drift_ann,
        "basis_mean_pct": w["basis"].mean() * 100, "basis_vol_ann_pct": basis_vol_ann * 100,
        "vol_ann": vol_ann, "mean_ann": mean_ann, "sharpe_gross": sharpe_gross,
    }
    for c in COSTS_BPS:
        cost_ann = (c / 10000.0) * (365.0 / ROLL_DAYS)
        out[f"sharpe_{int(c)}bps"] = (mean_ann - cost_ann) / vol_ann if vol_ann > 0 else float("nan")
    return out


def main():
    rc = sqlite3.connect(f"file:{RESEARCH_DB}?mode=ro", uri=True)
    pc = sqlite3.connect(f"file:{PROD_DB}?mode=ro", uri=True)

    structures = {
        "HL/CB":   lambda a: (_prices(rc, a, "hyperliquid", "perp"),
                              _prices(rc, a, "coinbase", "spot"), _hl_funding(rc, a)),
        "HL/BIN":  lambda a: (_prices(rc, a, "hyperliquid", "perp"),
                              _prices(rc, a, "binance", "spot"), _hl_funding(rc, a)),
        "CEX(bin)":lambda a: (_prices(rc, a, "binance", "perp"),
                              _prices(rc, a, "binance", "spot"), _binance_funding_hourly(pc, a)),
        "CEXx(b/cb)":lambda a: (_prices(rc, a, "binance", "perp"),
                              _prices(rc, a, "coinbase", "spot"), _binance_funding_hourly(pc, a)),
    }

    for struct, getter in structures.items():
        print("\n" + "=" * 122)
        print(f"STRUCTURE {struct}  (pnl = funding + spot_ret - perp_ret; hourly; Sharpe annualized x sqrt(8760))")
        print("=" * 122)
        print(f"{'asset':<6}{'window':<9}{'n':>6}{'fund_ann%':>10}{'basisMean%':>11}"
              f"{'basisVol_ann%':>14}{'pnl_vol%':>10}{'net_ann%':>10}"
              f"{'Sh_gross':>9}{'Sh_9':>7}{'Sh_20':>7}{'Sh_35':>7}")
        print("-" * 122)
        for a in ASSETS:
            perp, spot, funding = getter(a)
            df = _build(perp, spot, funding)
            if df.empty:
                print(f"{a:<6}  (insufficient overlap)")
                continue
            for wname, (lo, hi) in WINDOWS.items():
                m = _metrics(df, lo, hi)
                if m is None:
                    continue
                print(f"{a:<6}{wname:<9}{m['n']:>6}{m['funding_ann']*100:>10.2f}"
                      f"{m['basis_mean_pct']:>9.3f}{m['basis_vol_ann_pct']:>12.1f}"
                      f"{m['vol_ann']*100:>10.1f}{m['mean_ann']*100:>10.2f}"
                      f"{m['sharpe_gross']:>9.2f}{m['sharpe_9bps']:>7.2f}"
                      f"{m['sharpe_20bps']:>7.2f}{m['sharpe_35bps']:>7.2f}")
    # ---- THREE-WAY ANCHOR LADDER (BTC, ETH) over the COMMON (HL) window ----
    print("\n" + "=" * 100)
    print("THREE-WAY LADDER (BTC, ETH) over the COMMON window (= HL availability, ~7mo) "
          "-- attributes degradation")
    print("  A binance-perp / binance-spot   (same-venue: should ~reproduce idealized; basis~0)")
    print("  B binance-perp / coinbase-spot  (cross-CEX basis per se, NO DeFi)")
    print("  C HL-perp / coinbase-spot       (the real target; HL funding ~2x + HL basis)")
    print("=" * 100)
    print(f"{'asset':<6}{'leg':<24}{'n':>6}{'fund_ann%':>10}{'basisVol_ann%':>14}"
          f"{'Sh_gross':>10}{'Sh_9bps':>9}{'Sh_35bps':>9}")
    for a in ["BTC", "ETH"]:
        dfA = _build(_prices(rc, a, "binance", "perp"), _prices(rc, a, "binance", "spot"),
                     _binance_funding_hourly(pc, a))
        dfB = _build(_prices(rc, a, "binance", "perp"), _prices(rc, a, "coinbase", "spot"),
                     _binance_funding_hourly(pc, a))
        dfC = _build(_prices(rc, a, "hyperliquid", "perp"), _prices(rc, a, "coinbase", "spot"),
                     _hl_funding(rc, a))
        if dfC.empty:
            continue
        cstart = dfC.index.min()
        for label, df in [("A bin-perp/bin-spot", dfA), ("B bin-perp/cb-spot", dfB),
                          ("C HL-perp/cb-spot", dfC)]:
            m = _metrics(df, cstart, NOW)
            if m is None:
                print(f"{a:<6}{label:<24} (insufficient)")
                continue
            print(f"{a:<6}{label:<24}{m['n']:>6}{m['funding_ann']*100:>10.2f}"
                  f"{m['basis_vol_ann_pct']:>14.1f}{m['sharpe_gross']:>10.2f}"
                  f"{m['sharpe_9bps']:>9.2f}{m['sharpe_35bps']:>9.2f}")

    # ---- BREAKEVEN HOLD: days to clear RT cost at explicit execution assumptions ----
    print("\n" + "=" * 100)
    print("BREAKEVEN HOLD (days) = cost_RT_bps/10000 * 365 / funding_ann   (HL/CB, last-90d funding)")
    print("  cost RT:  9bps = HL taker perp (~9) + Coinbase MAKER spot (~0)  [OPTIMISTIC]")
    print("           35bps = HL taker + light Coinbase taker / slippage     [MIXED]")
    print("          100bps = HL taker + Coinbase TAKER spot (~40-60bps/side) [REALISTIC CEILING]")
    print("=" * 100)
    print(f"{'asset':<6}{'fund90d%':>10}{'minHold@9':>11}{'minHold@35':>12}{'minHold@100':>13}")
    lo90, hi90 = WINDOWS["last90d"]
    for a in ASSETS:
        dfC = _build(_prices(rc, a, "hyperliquid", "perp"), _prices(rc, a, "coinbase", "spot"),
                     _hl_funding(rc, a))
        if dfC.empty:
            continue
        m = _metrics(dfC, lo90, hi90)
        if m is None:
            continue
        f = m["funding_ann"]
        def mh(c, f=f):
            return "neg" if f <= 0 else f"{(c / 10000.0) * 365.0 / f:.1f}d"
        print(f"{a:<6}{f*100:>10.2f}{mh(9):>11}{mh(35):>12}{mh(100):>13}")

    rc.close(); pc.close()


if __name__ == "__main__":
    main()
