"""
engines/unlock_universe.py -- the unlock-bearing asset universe (Cycle 62A T3).

WHY THIS EXISTS
---------------
`market_data` held 427 rows across ADA / BTC / ETH / SOL / XRP: five mega-caps
whose supply moves by block subsidy, emission, burn and escrow. Forced-trade
scenario F1 is about tokens with VC and team VESTING CLIFFS, and that universe
contains none of them. Cycle 61 measured zero supply jumps clearing 1% across
it; the largest anywhere was 0.41%. F1 was unfalsifiable against that data --
not disconfirmed, unmeasurable.

WHY IT IS A SEPARATE LIST
-------------------------
The obvious move -- widening SUPPORTED_ASSETS in crypto_data_collector -- would
silently widen every other collector that iterates that dict (the market_data,
OHLCV and funding services all invoke `--asset all`), turning a 5-asset
CoinGecko cadence into a 25-asset one and changing existing collectors'
behaviour. The brief says additive only, so the unlock universe lives here, in
its own config-driven list, and existing collectors keep their scope exactly.

THE SELECTION RULE
------------------
Stated, mechanical and re-runnable, because a hand-picked list is not
auditable and this one has to be defensible later. Applied against a single
CoinGecko /coins/markets page:

  R1  LIQUIDITY FLOOR -- market_cap_rank <= --max-rank (default 300).
      An unlock in an illiquid token produces no measurable price response;
      below this floor F1's signal is unmeasurable rather than absent.

  R2  SUPPLY FIELDS PRESENT -- circulating_supply > 0 AND total_supply > 0.
      The brief's hard requirement. Checked here at selection time and AGAIN
      per asset against /coins/{id} by `verify`, because presence on the
      markets endpoint does not guarantee presence on the detail endpoint the
      collector actually reads.

  R3  LOCKED-SUPPLY OVERHANG -- circulating_supply / total_supply <= --max-float
      (default 0.75), i.e. at least 25% of supply is still locked.

  R4  NOT A STABLECOIN -- excluded via CoinGecko's own `stablecoins` category
      listing, fetched programmatically. Stablecoin supply moves by mint and
      redeem, which is not vesting. Excluded by rule, never by eye.

  R5  NOT ALREADY IN THE BASE UNIVERSE -- BTC/ETH/SOL/XRP/ADA/AVAX/BNB are
      already collected, and their locked supply is escrow, emission or
      staking rather than a vesting cliff. XRP passes R3 on 37% "locked"
      supply that is escrow release; admitting it would re-import exactly the
      mechanism Cycle 61 already measured as producing no jump above 0.41%.

  R6  TAKE THE TOP --size SURVIVORS by market cap (default 25).

ON "POST-2021 LAUNCHES"
-----------------------
The brief frames the target as post-2021 launches with large locked
allocations. R3 measures that property DIRECTLY rather than proxying it by
launch date, and is the better filter for F1: a 2019 token with 40% of supply
still locked can produce an unlock cliff, and a 2022 token that has fully
unlocked cannot. Launch date is a proxy for the overhang; the overhang itself
is observable, so it is what the rule uses.

Genesis dates are still fetched and recorded for every survivor so the
post-2021 question stays answerable from the output, and --min-genesis-year
applies it as a hard filter if wanted. It is OFF by default because CoinGecko
reports genesis_date as null for a large share of tokens, and a filter that
silently drops every asset with a missing field is exactly the kind of quiet
degradation Cycle 62A T5 exists to stamp out.

Usage:
    python -m engines.unlock_universe select     # run the rule, write config
    python -m engines.unlock_universe verify     # per-asset supply-field check
    python -m engines.unlock_universe show
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

logger = logging.getLogger("collector.unlock_universe")

REPO = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO / "config" / "unlock_universe.json"

CG_BASE = "https://api.coingecko.com/api/v3"

DEFAULT_MAX_RANK = 300
DEFAULT_MAX_FLOAT = 0.75
DEFAULT_SIZE = 25

# R5: assets the base collectors already cover. Their locked supply is escrow,
# emission or staking -- XRP's 37% "locked" is escrow release, not a VC cliff --
# which is the exact mechanism Cycle 61 measured and found produced no supply
# jump above 0.41%. Excluding them by rule keeps T3 additive to the existing
# universe rather than re-selecting it.
BASE_UNIVERSE = {"BTC", "ETH", "SOL", "XRP", "ADA", "AVAX", "BNB"}


def _get(path: str, params: dict, tries: int = 7) -> object:
    """CoinGecko GET with backoff. The free tier 429s readily, and its budget
    is shared across every call in the run -- so the retry ladder has to be
    patient enough to outlast a depleted window, not just a transient blip."""
    url = "%s/%s" % (CG_BASE, path.lstrip("/"))
    delay = 5.0
    for attempt in range(tries):
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 502, 503, 504):
            logger.warning("CoinGecko %s on %s, retrying in %.0fs",
                           r.status_code, path, delay)
            time.sleep(delay)
            delay *= 2
            continue
        raise RuntimeError("CoinGecko %s on %s: %s"
                           % (r.status_code, path, r.text[:200]))
    raise RuntimeError("CoinGecko exhausted retries on %s" % path)


def fetch_stablecoin_ids() -> set:
    """R4's exclusion set, from CoinGecko's own category -- not by eye."""
    out = set()
    for page in (1, 2):
        rows = _get("coins/markets", {
            "vs_currency": "usd", "category": "stablecoins",
            "order": "market_cap_desc", "per_page": 250, "page": page,
            "sparkline": "false",
        })
        if not rows:
            break
        out.update(c["id"] for c in rows)
        time.sleep(8.0)
    return out


def fetch_candidates(max_rank: int) -> list:
    """Top-ranked coins with the supply fields R2/R3 need."""
    rows = []
    per_page = 250
    for page in range(1, (max_rank // per_page) + 2):
        batch = _get("coins/markets", {
            "vs_currency": "usd", "order": "market_cap_desc",
            "per_page": per_page, "page": page, "sparkline": "false",
        })
        if not batch:
            break
        rows.extend(batch)
        if len(rows) >= max_rank:
            break
        time.sleep(8.0)
    return rows[:max_rank]


def apply_rule(rows: list, stablecoins: set, max_rank: int,
               max_float: float, size: int) -> tuple[list, dict]:
    """Run R1-R5. Returns (survivors, rejection tally)."""
    tally = {"r1_rank": 0, "r2_supply_missing": 0, "r3_float_too_high": 0,
             "r4_stablecoin": 0, "r5_base_universe": 0, "passed": 0}
    survivors = []

    for c in rows:
        rank = c.get("market_cap_rank")
        if rank is None or rank > max_rank:
            tally["r1_rank"] += 1
            continue
        if c.get("id") in stablecoins:
            tally["r4_stablecoin"] += 1
            continue
        circ = c.get("circulating_supply")
        total = c.get("total_supply")
        if not circ or not total or circ <= 0 or total <= 0:
            tally["r2_supply_missing"] += 1
            continue
        float_ratio = circ / total
        if float_ratio > max_float:
            tally["r3_float_too_high"] += 1
            continue
        if c["symbol"].upper() in BASE_UNIVERSE:
            tally["r5_base_universe"] += 1
            continue
        tally["passed"] += 1
        survivors.append({
            "asset": c["symbol"].upper(),
            "coingecko_id": c["id"],
            "name": c.get("name"),
            "market_cap_rank": rank,
            "market_cap_usd": c.get("market_cap"),
            "circulating_supply": circ,
            "total_supply": total,
            "max_supply": c.get("max_supply"),
            "float_ratio": round(float_ratio, 4),
            "locked_pct": round((1.0 - float_ratio) * 100, 2),
            "fully_diluted_valuation": c.get("fully_diluted_valuation"),
        })

    survivors.sort(key=lambda s: s["market_cap_rank"])
    return survivors[:size], tally


def enrich_genesis(survivors: list, sleep: float) -> None:
    """Attach genesis_date per survivor, for auditability of 'post-2021'."""
    for s in survivors:
        try:
            d = _get("coins/%s" % s["coingecko_id"], {
                "localization": "false", "tickers": "false",
                "market_data": "false", "community_data": "false",
                "developer_data": "false",
            })
            s["genesis_date"] = d.get("genesis_date")
        except Exception as e:
            logger.warning("genesis lookup failed for %s: %s", s["asset"], e)
            s["genesis_date"] = None
        time.sleep(sleep)


def load_universe() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            "%s does not exist -- run `python -m engines.unlock_universe "
            "select` first." % CONFIG_PATH)
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def universe_assets() -> list:
    """[(asset, coingecko_id)] for collectors. The only import other code needs."""
    u = load_universe()
    return [(a["asset"], a["coingecko_id"]) for a in u["assets"]]


# ----------------------------------------------------------------- select ---

def cmd_select(args) -> int:
    print("=" * 74)
    print("UNLOCK-BEARING UNIVERSE -- SELECTION")
    print("=" * 74)
    print("  R1 market_cap_rank <= %d" % args.max_rank)
    print("  R2 circulating_supply > 0 AND total_supply > 0")
    print("  R3 circulating_supply / total_supply <= %.2f "
          "(>= %.0f%% of supply still locked)"
          % (args.max_float, (1 - args.max_float) * 100))
    print("  R4 not in CoinGecko's stablecoins category")
    print("  R5 not already in the base universe %s"
          % ",".join(sorted(BASE_UNIVERSE)))
    print("  R6 top %d survivors by market cap" % args.size)
    if args.min_genesis_year:
        print("  R6 genesis_date year >= %d" % args.min_genesis_year)
    print("")

    try:
        stablecoins = fetch_stablecoin_ids()
        print("  fetched %d stablecoin ids for R4" % len(stablecoins))
        # The free-tier budget is shared across the whole run; let it refill
        # before the candidate pages rather than racing into a 429.
        time.sleep(args.phase_pause)
        rows = fetch_candidates(args.max_rank)
        print("  fetched %d ranked candidates" % len(rows))
    except Exception as e:
        print("\n[BLOCKED] CoinGecko could not serve the candidate set: %s" % e,
              file=sys.stderr)
        return 1

    survivors, tally = apply_rule(rows, stablecoins, args.max_rank,
                                  args.max_float, args.size)

    print("\n  -- rejections --")
    print("    R1 rank out of range   : %d" % tally["r1_rank"])
    print("    R2 supply field missing: %d" % tally["r2_supply_missing"])
    print("    R3 float ratio too high: %d" % tally["r3_float_too_high"])
    print("    R4 stablecoin          : %d" % tally["r4_stablecoin"])
    print("    R5 already in base universe: %d" % tally["r5_base_universe"])
    print("    passed                 : %d (taking top %d)"
          % (tally["passed"], args.size))

    if args.genesis:
        print("\n  fetching genesis dates for %d survivors..." % len(survivors))
        enrich_genesis(survivors, args.sleep)
        if args.min_genesis_year:
            before = len(survivors)
            survivors = [s for s in survivors if s.get("genesis_date")
                         and int(s["genesis_date"][:4]) >= args.min_genesis_year]
            print("  R6 genesis-year filter: %d -> %d"
                  % (before, len(survivors)))

    if len(survivors) < args.min_universe:
        print("\n[BLOCKED] the rule yielded %d assets, below the required "
              "minimum of %d. Reporting rather than loosening the rule to "
              "reach a number -- a rule tuned until it produces the desired "
              "count is not a rule."
              % (len(survivors), args.min_universe), file=sys.stderr)
        return 1

    print("\n  -- SELECTED UNIVERSE (%d assets) --" % len(survivors))
    print("    %-7s %-5s %-13s %-9s %-8s %s"
          % ("asset", "rank", "mcap($M)", "locked%", "genesis", "name"))
    for s in survivors:
        print("    %-7s %-5d %-13s %-9s %-8s %s"
              % (s["asset"], s["market_cap_rank"],
                 "%.0f" % ((s["market_cap_usd"] or 0) / 1e6),
                 "%.1f" % s["locked_pct"],
                 (s.get("genesis_date") or "-")[:7],
                 (s.get("name") or "")[:26]))

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Forced-trade scenario F1: tokens carrying VC/team vesting "
                   "cliffs, whose supply can jump at an unlock.",
        "selection_rule": {
            "R1_max_market_cap_rank": args.max_rank,
            "R2_requires": "circulating_supply > 0 AND total_supply > 0",
            "R3_max_float_ratio": args.max_float,
            "R4_excludes": "CoinGecko category=stablecoins",
            "R5_excludes_base_universe": sorted(BASE_UNIVERSE),
            "R6_size": args.size,
            "R7_min_genesis_year": args.min_genesis_year,
            "source": "CoinGecko /coins/markets",
            "note": "R3 measures locked-supply overhang directly rather than "
                    "proxying it by launch date; see module docstring.",
        },
        "rejections": tally,
        "assets": survivors,
    }
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n  wrote %s" % CONFIG_PATH)
    return 0


# ----------------------------------------------------------------- verify ---

def cmd_verify(args) -> int:
    """Per asset, against the endpoint the COLLECTOR uses, not the one the
    selector used. The brief requires verification, not assumption."""
    u = load_universe()
    print("=" * 74)
    print("UNLOCK UNIVERSE -- PER-ASSET SUPPLY-FIELD VERIFICATION")
    print("=" * 74)
    print("  endpoint: /coins/{id} (the one collect_unlock_market_data reads)")
    print("")
    ok, bad = [], []
    for a in u["assets"]:
        try:
            d = _get("coins/%s" % a["coingecko_id"], {
                "localization": "false", "tickers": "false",
                "community_data": "false", "developer_data": "false",
            })
            md = d.get("market_data") or {}
            circ = md.get("circulating_supply")
            total = md.get("total_supply")
            good = bool(circ) and bool(total) and circ > 0 and total > 0
            print("  [%s] %-7s circ=%-18s total=%-18s"
                  % ("OK  " if good else "FAIL", a["asset"], circ, total))
            (ok if good else bad).append(a["asset"])
        except Exception as e:
            print("  [FAIL] %-7s lookup error: %s" % (a["asset"], str(e)[:70]))
            bad.append(a["asset"])
        time.sleep(args.sleep)

    print("\n  %d/%d expose both circulating_supply and total_supply"
          % (len(ok), len(ok) + len(bad)))
    if bad:
        print("\n[FAIL] these do not serve F1 and must not be collected as if "
              "they did: %s" % ", ".join(bad), file=sys.stderr)
        return 1
    print("\n[OK] every selected asset exposes both supply fields.")
    return 0


def cmd_show(args) -> int:
    u = load_universe()
    print("generated_at: %s" % u["generated_at"])
    print("rule: %s" % json.dumps(u["selection_rule"], indent=2))
    print("\n%d assets:" % len(u["assets"]))
    for a in u["assets"]:
        print("  %-7s rank=%-4d locked=%5.1f%%  %s"
              % (a["asset"], a["market_cap_rank"], a["locked_pct"],
                 a["coingecko_id"]))
    return 0


def main():
    p = argparse.ArgumentParser(description="Unlock-bearing universe selection.")
    p.add_argument("--verbose", type=int, default=3, choices=[0, 1, 2, 3])
    p.add_argument("--sleep", type=float, default=6.5,
                   help="Seconds between CoinGecko detail calls (free tier "
                        "rate-limits aggressively; default 6.5)")
    subs = p.add_subparsers(dest="command", required=True)

    s = subs.add_parser("select", help="Run the rule and write the config")
    s.add_argument("--max-rank", type=int, default=DEFAULT_MAX_RANK)
    s.add_argument("--max-float", type=float, default=DEFAULT_MAX_FLOAT)
    s.add_argument("--size", type=int, default=DEFAULT_SIZE)
    s.add_argument("--phase-pause", type=float, default=20.0,
                   help="Seconds to idle between the stablecoin and candidate "
                        "fetches so the free-tier budget refills (default 20)")
    s.add_argument("--min-universe", type=int, default=20,
                   help="Fail rather than deliver a universe smaller than "
                        "this (default 20, the brief's floor)")
    s.add_argument("--genesis", action="store_true", default=True,
                   help="Fetch genesis_date per survivor (default on)")
    s.add_argument("--no-genesis", dest="genesis", action="store_false")
    s.add_argument("--min-genesis-year", type=int, default=None,
                   help="Hard-filter on launch year. OFF by default: "
                        "CoinGecko reports genesis_date as null for many "
                        "tokens, and a filter that silently drops every asset "
                        "with a missing field is the degradation Cycle 62A T5 "
                        "exists to prevent.")

    v = subs.add_parser("verify", help="Per-asset supply-field verification")
    subs.add_parser("show", help="Print the current universe")

    args = p.parse_args()
    logging.basicConfig(
        level={0: logging.ERROR, 1: logging.WARNING, 2: logging.INFO,
               3: logging.DEBUG}.get(args.verbose, logging.DEBUG),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")

    if args.command == "select":
        return cmd_select(args)
    if args.command == "verify":
        return cmd_verify(args)
    if args.command == "show":
        return cmd_show(args)
    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
