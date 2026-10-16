"""
engines/forced_trade/leveraged_class.py

Cycle 62B T4 -- re-measure, against the real archive listing, what the Cycle 60
leveraged-token SUBSTRING rule caught, now that universe.py identifies
leveraged tokens structurally (underlying must trade).

The bug is a CLASS: any base name ending in UP/DOWN/BULL/BEAR is caught by the
substring rule (and by an endswith rule -- JUPUSDT ends with UPUSDT). So this
reports the whole class, split into genuine leveraged tokens and ordinary spot
assets the old rule deleted, rather than restating Cycle 61's two names.

Run:
    python -m engines.forced_trade.leveraged_class \
        --symbols-file claude/scratch/c62b_archive_symbols_raw.txt
(omit --symbols-file to enumerate the S3 bucket listing fresh; needs network)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from engines.xsec_reversal.archive import ArchiveClient
from engines.xsec_reversal.universe import (
    DEFAULT_LEVERAGED_PATTERNS, UniverseSpec, filter_symbol_names,
    is_leveraged_token, leveraged_underlying,
)

C61_CANDIDATES = Path("outputs/forced_trade/t3_candidates.csv")


def substring_caught(symbols, patterns=DEFAULT_LEVERAGED_PATTERNS) -> list:
    """What the Cycle 60 rule removed: any legacy pattern `in` the name."""
    return sorted(s for s in symbols if any(p in s.upper() for p in patterns))


def classify_class(symbols, spec: UniverseSpec) -> pd.DataFrame:
    """Every name that SPELLS as a leveraged token, with the structural verdict."""
    traded = {s.upper() for s in symbols}
    rows = []
    for s in sorted(traded):
        if spec.quote and not s.endswith(spec.quote):
            continue
        und = leveraged_underlying(s, spec)
        if und is None:
            continue
        rows.append({"symbol": s, "implied_underlying": und,
                     "underlying_trades": und in traded,
                     "leveraged": is_leveraged_token(s, traded, spec)})
    return pd.DataFrame(rows)


def months_in_window(client, symbol, interval, start, end) -> dict:
    """Archive months a symbol has inside [start, end] (YYYY-MM strings)."""
    periods = client.list_symbol_periods(symbol, interval)
    inside = [p for p in periods if start <= p <= end]
    return {"symbol": symbol, "first": periods[0] if periods else None,
            "last": periods[-1] if periods else None,
            "months_in_window": len(inside)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("--symbols-file", default=None,
                    help="raw archive listing (all quotes); omit to enumerate S3")
    ap.add_argument("--quote", default="USDT", help="Cycle 60 study quote")
    ap.add_argument("--interval", default="4h", help="Cycle 60 bar interval")
    ap.add_argument("--window-start", default="2021-01", help="Cycle 60 panel start")
    ap.add_argument("--window-end", default="2026-06", help="Cycle 60 panel end")
    ap.add_argument("--cache-dir", default="data/external/binance_archive")
    ap.add_argument("--out", default="outputs/forced_trade/c62b/t4_leveraged_class.json")
    args = ap.parse_args(argv)

    client = ArchiveClient(cache_dir=Path(args.cache_dir))
    if args.symbols_file:
        raw = Path(args.symbols_file).read_text(encoding="utf-8")
        listing = sorted({s.strip() for s in raw.split() if s.strip()})
        source = args.symbols_file
    else:
        listing = client.list_all_symbols(quote=None)
        source = "S3 bucket listing (fresh)"
    q_syms = [s for s in listing if s.endswith(args.quote)]
    print(f"listing: {len(listing)} symbols, all quotes, INCLUDING DELISTED ({source})")
    print(f"{args.quote}-quoted: {len(q_syms)}")

    spec = UniverseSpec(quote=args.quote)
    caught = substring_caught(q_syms)
    lev = sorted(s for s in q_syms if is_leveraged_token(s, set(listing), spec))
    not_caught_but_lev = sorted(set(lev) - set(caught))
    restored = sorted(set(caught) - set(lev))
    print(f"\n[{args.quote}] Cycle 60 substring rule catches: {len(caught)}")
    print(f"  genuine leveraged tokens (underlying trades): {len(lev)}")
    print(f"  ordinary spot assets it deleted: {len(restored)} -> {restored}")
    if not_caught_but_lev:
        print(f"  !! structural-only catches (should be empty): {not_caught_but_lev}")

    # The whole spelling class, every quote: names that PARSE as leveraged.
    all_cls = classify_class(listing, UniverseSpec(quote=""))
    lookalikes_all = all_cls.loc[~all_cls["leveraged"], "symbol"].tolist()
    print(f"\n[all quotes] names spelled <BASE><UP|DOWN|BULL|BEAR><USDT|BUSD|USDC>: "
          f"{len(all_cls)}; genuine {int(all_cls['leveraged'].sum())}; "
          f"lookalikes {len(lookalikes_all)} -> {lookalikes_all}")

    # Full Cycle 60 name filter, old rule vs new rule, same listing.
    new_kept = filter_symbol_names(q_syms, spec, traded=listing)
    old_kept = [s for s in filter_symbol_names(q_syms, UniverseSpec(
        quote=args.quote, exclude_leveraged=False), traded=listing)
        if not any(p in s for p in DEFAULT_LEVERAGED_PATTERNS)]
    gained = sorted(set(new_kept) - set(old_kept))
    lost = sorted(set(old_kept) - set(new_kept))
    print(f"\nfull name filter on today's listing: old rule keeps {len(old_kept)}, "
          f"new rule keeps {len(new_kept)} (+{len(gained)} {gained}, -{len(lost)} {lost})")

    # Did the structural rule label anything leveraged that Cycle 61 did not?
    drift = None
    if C61_CANDIDATES.exists():
        c61 = pd.read_csv(C61_CANDIDATES)
        c61_gen = set(c61.loc[c61["is_genuine"] & (c61["quote"] == args.quote), "symbol"])
        drift = {"cycle61_genuine": len(c61_gen),
                 "new_genuine_not_in_c61": sorted(set(lev) - c61_gen),
                 "c61_genuine_not_now": sorted(c61_gen - set(lev))}
        print(f"\nvs Cycle 61 genuine {args.quote} list ({len(c61_gen)}): "
              f"new={drift['new_genuine_not_in_c61']} gone={drift['c61_genuine_not_now']}")

    # Impact on Cycle 60: archive months each restored name has in its window.
    cov = [months_in_window(client, s, args.interval, args.window_start, args.window_end)
           for s in restored]
    print(f"\nrestored names' {args.interval} archive months inside the Cycle 60 window "
          f"{args.window_start}..{args.window_end}:")
    for c in cov:
        print(f"  {c['symbol']:<12} first={c['first']} last={c['last']} "
              f"months_in_window={c['months_in_window']}")

    result = {
        "source": source, "n_listing": len(listing), "quote": args.quote,
        "n_quote_symbols": len(q_syms),
        "substring_caught": caught, "genuine_leveraged": lev,
        "restored_spot_assets": restored,
        "structural_only_catches": not_caught_but_lev,
        "all_quotes_spelled_class": len(all_cls),
        "all_quotes_lookalikes": lookalikes_all,
        "name_filter_old_kept": len(old_kept), "name_filter_new_kept": len(new_kept),
        "name_filter_gained": gained, "name_filter_lost": lost,
        "cycle61_drift": drift, "restored_coverage": cov,
        "cycle60_window": [args.window_start, args.window_end],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
