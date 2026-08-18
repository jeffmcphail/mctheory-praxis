"""Build the aligned GLD/GDX minute panel and persist it as parquet.

Parsing ~600 MB of Kibot text on every backtest run is wasteful; the parquet
loads in seconds. The raw .txt files stay the source of truth.

The panel deliberately joins BOTH delivered layouts, because neither one is
sufficient on its own:

  * the 10-column bid/ask files carry quotes but no volume and no trade price
  * the 7-column OHLCV files carry trade prices and volume but no quotes

Join policy:
  * index = timestamps where BOTH symbols have quotes (inner join) -- you
    cannot trade the pair on a bar where one leg has no quote
  * trade OHLCV is LEFT-joined onto that index, so a minute that quoted but
    never printed keeps its quote row with NaN trade fields. That sparsity is
    information about liquidity, not missing data to be dropped.
  * bars carrying a non-positive price are dropped outright (vendor
    corruption; all 7 observed cases are 04:00 zero-ask prints)

    python -m engines.chan_cpo.build_panel --adjustment adj
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from engines.chan_cpo.data_loader import (
    DEFAULT_BIDASK_SCHEMA,
    DEFAULT_OHLCV_SCHEMA,
    load_kibot,
)

DATA_DIR = Path("data/external/kibot")
QUOTE_PRICE_COLS = (
    "bid_open", "bid_high", "bid_low", "bid_close",
    "ask_open", "ask_high", "ask_low", "ask_close",
)


def resolve(symbol: str, kind: str, adjustment: str, data_dir: Path) -> Path:
    """Find a delivery file despite Kibot's inconsistent separators.

    The 2026-08 delivery names the GLD OHLCV files `GLD.1m_adj.txt` (dot) but
    everything else `GDX_1m_adj.txt` (underscore), so try both.
    """
    stem = f"1m_bidask_{adjustment}" if kind == "bidask" else f"1m_{adjustment}"
    candidates = [data_dir / f"{symbol}{sep}{stem}.txt" for sep in ("_", ".")]
    for path in candidates:
        if path.exists():
            return path
    tried = ", ".join(p.name for p in candidates)
    raise FileNotFoundError(f"no {symbol} {kind} {adjustment} file (tried {tried})")


def build_panel(adjustment: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    quotes: dict[str, pd.DataFrame] = {}
    trades: dict[str, pd.DataFrame] = {}

    for sym in ("GLD", "GDX"):
        q = load_kibot(resolve(sym, "bidask", adjustment, data_dir),
                       DEFAULT_BIDASK_SCHEMA)
        bad = (q[list(QUOTE_PRICE_COLS)] <= 0).any(axis=1)
        if bad.any():
            print(f"  {sym}: dropping {int(bad.sum())} bars with non-positive price")
            q = q[~bad]
        q["crossed"] = q["bid_low"] > q["ask_high"]
        q["locked"] = q["spread"].abs() < 1e-9
        quotes[sym] = q

        t = load_kibot(resolve(sym, "ohlcv", adjustment, data_dir),
                       DEFAULT_OHLCV_SCHEMA)
        trades[sym] = t.rename(columns={c: f"trade_{c}" for c in t.columns})
        print(f"  {sym}: {len(q):,} quote bars, {len(t):,} trade bars")

    common = quotes["GLD"].index.intersection(quotes["GDX"].index)
    print(f"  pair-common quote bars: {len(common):,}")

    frames = []
    for sym in ("GLD", "GDX"):
        joined = quotes[sym].loc[common].join(trades[sym], how="left")
        frames.append(joined.add_prefix(f"{sym.lower()}_"))
    panel = pd.concat(frames, axis=1)
    panel.index.name = "timestamp"
    return panel


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adjustment", default="adj", choices=["adj", "unadj"])
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--compression", default="zstd")
    args = ap.parse_args(argv)

    print(f"=== building pair panel: adjustment={args.adjustment} ===")
    panel = build_panel(args.adjustment, args.data_dir)

    out = args.out or (
        args.data_dir / f"pair_gld_gdx_1m_{args.adjustment}.parquet"
    )
    panel.to_parquet(out, compression=args.compression, index=True)

    size_mb = out.stat().st_size / 1e6
    trade_cov = 100.0 * panel["gld_trade_close"].notna().mean()
    print(f"\nwrote {out}  ({size_mb:.1f} MB, {args.compression})")
    print(f"  rows: {len(panel):,}   columns: {len(panel.columns)}")
    print(f"  range: {panel.index[0]} .. {panel.index[-1]}")
    print(f"  bars with a GLD trade print: {trade_cov:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
