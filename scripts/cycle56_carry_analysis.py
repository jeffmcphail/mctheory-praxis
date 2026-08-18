"""
scripts/cycle56_carry_analysis.py
=================================
Cycle 56 RECON — first-pass carry analysis (V1 ranking + V3 cost gate).

Reads DeFi funding from the isolated research DB (hyperliquid, dydx; HOURLY)
and CEX funding from production crypto_data.db (binance, bybit; 8-HOURLY), over
a uniform window (2024-01-01 -> now), and computes per (venue, asset):
  - n observations
  - mean periodic funding rate
  - mean ANNUALIZED funding (cadence-correct: hourly ×8760, 8h ×1095)
  - positive_share (fraction of periods with rate > 0 = short earns)
  - mean annualized funding conditional on rate>0 (selective-short upper bound)
  - min hold-days to clear a round-trip cost C (the V3 gate metric)

NO network, NO funds — pure read + arithmetic over already-collected data.

THE GATE METRIC — min_hold_days:
  A delta-neutral short carry pays the all-in round-trip cost C once per hold.
  Annualized cost = C × (365 / H). Net carry > 0  <=>  mean_ann > C × 365/H
  <=>  H > C × 365 / mean_ann.  So min_hold_days = C_frac × 365 / mean_ann_frac.
  Low-funding majors need long holds; high-funding assets clear at short holds.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROD_DB = REPO / "data" / "crypto_data.db"
RESEARCH_DB = REPO / "data" / "defi_funding_research.db"

WINDOW_START_MS = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

# periods/year by cadence
ANN = {"hourly": 24 * 365, "8h": 3 * 365}

# Representative all-in round-trip costs (perp leg + spot hedge leg), bps:
COSTS_BPS = {
    "optimistic (HL maker + CB maker)": 15.0,
    "realistic (HL taker + Kraken taker)": 35.0,
}


def cadence_for(venue: str) -> str:
    return "8h" if venue in ("binance", "bybit") else "hourly"


def load(db: Path, venues: list[str]) -> dict:
    """Return {(venue,asset): [rates...]} for rows in the window."""
    out: dict = {}
    if not db.exists():
        return out
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        q = ("SELECT venue, asset, funding_rate FROM funding_rates "
             "WHERE timestamp >= ? AND venue IN (%s) ORDER BY asset"
             % ",".join("?" * len(venues)))
        for v, a, r in conn.execute(q, (WINDOW_START_MS, *venues)):
            out.setdefault((v, a), []).append(float(r))
    finally:
        conn.close()
    return out


def stats(venue: str, rates: list[float]) -> dict:
    n = len(rates)
    if n == 0:
        return {}
    f = ANN[cadence_for(venue)]
    mean = sum(rates) / n
    pos = [r for r in rates if r > 0]
    pos_share = len(pos) / n
    mean_pos = (sum(pos) / len(pos)) if pos else 0.0
    mean_ann = mean * f
    mean_ann_pos = mean_pos * f
    return {
        "n": n, "mean": mean, "mean_ann": mean_ann,
        "pos_share": pos_share, "mean_ann_pos": mean_ann_pos,
    }


def min_hold_days(mean_ann_frac: float, cost_bps: float) -> float:
    if mean_ann_frac <= 0:
        return float("inf")
    return (cost_bps / 10000.0) * 365.0 / mean_ann_frac


def main():
    data = {}
    data.update(load(RESEARCH_DB, ["hyperliquid", "dydx"]))
    data.update(load(PROD_DB, ["binance", "bybit"]))

    rows = []
    for (venue, asset), rates in data.items():
        s = stats(venue, rates)
        if not s:
            continue
        rows.append((venue, asset, s))

    # sort by annualized funding desc
    rows.sort(key=lambda x: x[2]["mean_ann"], reverse=True)

    print("=" * 110)
    print("CYCLE 56 CARRY ANALYSIS — window 2024-01-01 -> now")
    print("annualized = cadence-correct (hourly x8760, 8h x1095). "
          "mean_ann = always-short; pos = selective (rate>0 only)")
    print("=" * 110)
    hdr = (f"{'venue':<12}{'asset':<8}{'n':>7}{'mean_ann%':>11}"
           f"{'pos_share':>11}{'mean_ann_pos%':>15}"
           f"{'minHold@15bps':>15}{'minHold@35bps':>15}")
    print(hdr)
    print("-" * 110)
    for venue, asset, s in rows:
        mh15 = min_hold_days(s["mean_ann"], 15.0)
        mh35 = min_hold_days(s["mean_ann"], 35.0)
        def fmt(d):
            return "neg/never" if d == float("inf") else f"{d:.1f}d"
        print(f"{venue:<12}{asset:<8}{s['n']:>7}{s['mean_ann']*100:>11.2f}"
              f"{s['pos_share']:>11.2f}{s['mean_ann_pos']*100:>15.2f}"
              f"{fmt(mh15):>15}{fmt(mh35):>15}")

    # ---- Majors cross-venue anchor (V3a) ----
    print("\n" + "=" * 70)
    print("V3a ANCHOR — majors cross-venue mean annualized funding (%)")
    print("=" * 70)
    majors = ["BTC", "ETH", "SOL", "XRP", "ADA", "AVAX"]
    venues = ["binance", "bybit", "hyperliquid", "dydx"]
    print(f"{'asset':<8}" + "".join(f"{v:>13}" for v in venues))
    for a in majors:
        cells = []
        for v in venues:
            s = stats(v, data.get((v, a), []))
            cells.append(f"{s['mean_ann']*100:>13.2f}" if s else f"{'—':>13}")
        print(f"{a:<8}" + "".join(cells))

    # ---- Gate summary ----
    print("\n" + "=" * 70)
    print("V3 GATE — assets clearing a 35bps round-trip at a <=14d hold")
    print("(mean_ann always-short; selective timing would be easier)")
    print("=" * 70)
    clears = [(v, a, s) for (v, a, s) in rows
              if min_hold_days(s["mean_ann"], 35.0) <= 14.0]
    if not clears:
        print("  NONE clear at <=14d on always-short mean. Check selective/pos.")
    for venue, asset, s in clears:
        print(f"  {venue:<12}{asset:<8} mean_ann={s['mean_ann']*100:6.2f}%  "
              f"minHold@35bps={min_hold_days(s['mean_ann'],35.0):.1f}d  "
              f"pos_share={s['pos_share']:.2f}")


if __name__ == "__main__":
    main()
