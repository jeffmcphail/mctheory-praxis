"""
portfolio.py — Polymarket Portfolio Dashboard v4

Unified view: merges Data API positions with CLOB open orders
to show each trade once with filled + pending status.

Usage:
    python portfolio.py                       # Full dashboard
    python portfolio.py --summary             # Summary only
    python portfolio.py --grep "tel aviv"     # Search
    python portfolio.py --grep "CPI"          # Search
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv()

import requests

HOST = "https://clob.polymarket.com"
DATA_API = "https://data-api.polymarket.com"
CHAIN_ID = 137


def get_client():
    from py_clob_client.client import ClobClient
    pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    client = ClobClient(HOST, key=pk, chain_id=CHAIN_ID, signature_type=0)
    client.set_api_creds(client.derive_api_key())
    return client


def get_positions(address):
    r = requests.get(f"{DATA_API}/positions", params={"user": address}, timeout=10)
    return r.json() if r.ok else []


def get_recent_trades(address, limit=50):
    r = requests.get(f"{DATA_API}/trades",
                     params={"user": address, "limit": str(limit)}, timeout=10)
    return r.json() if r.ok else []


def get_open_orders(client):
    try:
        orders = client.get_orders()
        return orders if isinstance(orders, list) else []
    except:
        return []


def classify(title, slug=""):
    t = (title + " " + slug).lower()
    if any(w in t for w in ["temperature", "weather", "°c", "°f"]):
        return "weather"
    elif any(w in t for w in ["inflation", "cpi", "gdp", "rate", "rbi"]):
        return "macro"
    elif any(w in t for w in ["nba", "nfl", "mlb", "nhl", "win", "playoff",
                               "seed", "division", "regular season"]):
        return "sports"
    elif any(w in t for w in ["election", "president", "governor", "party"]):
        return "politics"
    return "other"


def lookup_market_titles(orphan_orders):
    """Look up market titles for orders without positions via Gamma API."""
    titles = {}  # asset_id -> {title, slug, end_date}

    for o in orphan_orders:
        asset = o.get("asset_id", "")
        if not asset or asset in titles:
            continue
        try:
            r = requests.get("https://gamma-api.polymarket.com/markets",
                             params={"clob_token_ids": asset}, timeout=5)
            if r.ok:
                data = r.json()
                if isinstance(data, list) and len(data) == 1:
                    m = data[0]
                    titles[asset] = {
                        "title": m.get("question", m.get("groupItemTitle", "?")),
                        "slug": m.get("slug", ""),
                        "end_date": (m.get("endDate") or "")[:10],
                    }
        except Exception:
            pass
        time.sleep(0.15)

    return titles


def build_unified_trades(positions, open_orders):
    """
    Merge positions (filled shares) with open orders (unfilled shares)
    into a single unified trade list. Match by asset_id.
    """
    # Index positions by asset ID
    pos_by_asset = {}
    for p in positions:
        asset = p.get("asset", "")
        pos_by_asset[asset] = p

    # Index open orders by asset ID
    orders_by_asset = {}
    for o in open_orders:
        asset = o.get("asset_id", "")
        if asset not in orders_by_asset:
            orders_by_asset[asset] = []
        orders_by_asset[asset].append(o)

    # All unique asset IDs
    all_assets = set(list(pos_by_asset.keys()) + list(orders_by_asset.keys()))

    # Find orphan assets (have orders but no position) and look up titles
    orphan_orders = []
    for asset in all_assets:
        if asset not in pos_by_asset and asset in orders_by_asset:
            orphan_orders.extend(orders_by_asset[asset])

    gamma_titles = {}
    if orphan_orders:
        gamma_titles = lookup_market_titles(orphan_orders)

    unified = []
    for asset in all_assets:
        pos = pos_by_asset.get(asset)
        orders = orders_by_asset.get(asset, [])

        # Get title from position, then Gamma API as fallback
        title = ""
        slug = ""
        end_date = ""
        outcome = ""
        cur_price = 0

        if pos:
            title = pos.get("title", "?")
            slug = pos.get("slug", "")
            end_date = pos.get("endDate", "")[:10]
            outcome = pos.get("outcome", "")
            cur_price = float(pos.get("curPrice", 0))
        elif asset in gamma_titles:
            title = gamma_titles[asset].get("title", "")
            slug = gamma_titles[asset].get("slug", "")
            end_date = gamma_titles[asset].get("end_date", "")

        # Filled portion
        filled_shares = float(pos.get("size", 0)) if pos else 0
        avg_price = float(pos.get("avgPrice", 0)) if pos else 0
        cost_basis = float(pos.get("initialValue", 0)) if pos else 0
        current_value = float(pos.get("currentValue", 0)) if pos else 0
        cash_pnl = float(pos.get("cashPnl", 0)) if pos else 0
        redeemable = pos.get("redeemable", False) if pos else False

        # Open order portion
        pending_shares = 0
        pending_capital = 0
        order_price = 0
        order_ids = []

        for o in orders:
            orig = float(o.get("original_size", 0))
            matched = float(o.get("size_matched", 0))
            remaining = orig - matched
            price = float(o.get("price", 0))
            pending_shares += remaining
            pending_capital += remaining * price
            order_price = price
            order_ids.append(o.get("id", "")[:16])

        total_shares = filled_shares + pending_shares
        total_committed = cost_basis + pending_capital

        # Calculate fill percentage
        if total_shares > 0:
            fill_pct = filled_shares / total_shares
        else:
            fill_pct = 0

        # Determine status
        if filled_shares > 0 and pending_shares == 0:
            status = "FILLED"
        elif filled_shares > 0 and pending_shares > 0:
            status = "PARTIAL"
        elif pending_shares > 0:
            status = "PENDING"
        else:
            status = "?"

        # P&L calculation (our own, not the API's weird percentages)
        if cost_basis > 0:
            pnl_pct = cash_pnl / cost_basis
        else:
            pnl_pct = 0

        category = classify(title, slug)

        unified.append({
            "asset": asset,
            "title": title,
            "slug": slug,
            "outcome": outcome,
            "end_date": end_date,
            "category": category,
            "cur_price": cur_price,
            "redeemable": redeemable,
            # Filled
            "filled_shares": filled_shares,
            "avg_price": avg_price,
            "cost_basis": cost_basis,
            "current_value": current_value,
            "cash_pnl": cash_pnl,
            "pnl_pct": pnl_pct,
            # Pending
            "pending_shares": pending_shares,
            "pending_capital": pending_capital,
            "order_price": order_price,
            "order_ids": order_ids,
            # Totals
            "total_shares": total_shares,
            "total_committed": total_committed,
            "fill_pct": fill_pct,
            "status": status,
        })

    # Sort by total committed descending
    unified.sort(key=lambda u: u["total_committed"], reverse=True)
    return unified


def main():
    parser = argparse.ArgumentParser(description="Portfolio Dashboard v4")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--grep", type=str, default=None)
    args = parser.parse_args()

    address = os.getenv("POLYMARKET_API_ADDRESS", "")

    print(f"\n{'='*143}")
    print(f"POLYMARKET PORTFOLIO DASHBOARD v4 — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Wallet: {address[:10]}...{address[-6:]}")
    print(f"{'='*143}")

    print(f"\n  Loading data...")
    client = get_client()
    positions = get_positions(address)
    open_orders = get_open_orders(client)
    recent_trades = get_recent_trades(address)

    unified = build_unified_trades(positions, open_orders)

    print(f"  {len(positions)} positions | {len(open_orders)} open orders | "
          f"{len(unified)} unique trades")

    # Apply grep
    if args.grep:
        p = args.grep.lower()
        unified = [u for u in unified if p in u.get("title", "").lower()
                   or p in u.get("slug", "").lower()
                   or p in u.get("category", "").lower()]

    # === UNIFIED TRADES TABLE ===
    if not args.summary:
        hdr = (f"  {'#':<3s} {'Title':<80s} {'Side':<5s} {'Cat':<8s} {'Resolve':<10s}"
               f" {'Entry':>6s} {'CurPx':>6s} {'Filled':>8s} {'Pend':>8s} {'Status':>9s}"
               f" {'Cost':>9s} {'Value':>9s} {'P&L':>9s} {'P&L%':>7s}")
        W = len(hdr)

        print(f"\n{'─'*W}")
        print(f"  ALL TRADES ({len(unified)})")
        print(f"{'─'*W}")
        print(hdr)
        print(f"{'─'*W}")

        for i, u in enumerate(unified, 1):
            title = u["title"][:79] if u["title"] else f"(asset {u['asset'][:12]}...)"
            side = u.get("outcome", "?")[:4]
            cat = u["category"][:8]
            end = u["end_date"]
            filled = u["filled_shares"]
            pending = u["pending_shares"]
            fill_pct = u["fill_pct"]
            status = u["status"]
            cur_price = u.get("cur_price", 0)
            redeem = " 🎯" if u["redeemable"] else ""

            # Entry price: avg_price for filled, order_price for pending
            entry = u["avg_price"] if u["avg_price"] > 0 else u.get("order_price", 0)

            # Current price string
            cur_str = f"{cur_price:>6.3f}" if cur_price > 0 else f"{'—':>6s}"

            # Combined fill+status: "100% FILL", " 77% PART", "  0% PEND"
            short = {"FILLED": "FILL", "PARTIAL": "PART", "PENDING": "PEND"}.get(status, status[:4])
            fill_status = f"{fill_pct:>4.0%} {short:<4s}"

            # Cost/Value/P&L
            if status == "PENDING":
                cost_str = f"${u.get('pending_capital', 0):>8.2f}"
                val_str  = f"{'—':>9s}"
                pnl_str  = f"{'—':>9s}"
                pct_str  = f"{'—':>7s}"
            else:
                cost = u["cost_basis"]
                value = u["current_value"]
                pnl = u["cash_pnl"]
                pnl_pct = u["pnl_pct"]
                cost_str = f"${cost:>8.2f}"
                val_str  = f"${value:>8.2f}"
                pnl_str  = f"+${pnl:>7.2f}" if pnl >= 0 else f"-${abs(pnl):>7.2f}"
                pct_str  = (f"+{pnl_pct*100:>5.1f}%" if pnl_pct >= 0 else f"{pnl_pct*100:>6.1f}%") if cost > 0 else f"{'—':>7s}"

            print(f"  {i:<3d} {title:<80s} {side:<5s} {cat:<8s} {end:<10s}"
                  f" {entry:>6.3f} {cur_str} {filled:>8.1f} {pending:>8.1f} {fill_status}"
                  f" {cost_str} {val_str} {pnl_str} {pct_str}{redeem}")

        # Totals
        t_filled = sum(u["filled_shares"] for u in unified)
        t_pending = sum(u["pending_shares"] for u in unified)
        t_cost = sum(u["cost_basis"] for u in unified)
        t_value = sum(u["current_value"] for u in unified)
        t_pnl = sum(u["cash_pnl"] for u in unified)
        t_pnl_pct = t_pnl / t_cost if t_cost > 0 else 0

        t_pnl_str = f"+${t_pnl:>7.2f}" if t_pnl >= 0 else f"-${abs(t_pnl):>7.2f}"
        t_pct_str = f"+{t_pnl_pct*100:>5.1f}%" if t_pnl_pct >= 0 else f"{t_pnl_pct*100:>6.1f}%"

        print(f"{'─'*W}")
        print(f"  {'TOT':<3s} {'':80s} {'':5s} {'':8s} {'':10s}"
              f" {'':>6s} {'':>6s} {t_filled:>8.1f} {t_pending:>8.1f} {'':>9s}"
              f" ${t_cost:>8.2f} ${t_value:>8.2f} {t_pnl_str} {t_pct_str}")

    # === SUMMARY ===
    print(f"\n{'='*143}")
    print(f"  PORTFOLIO SUMMARY")
    print(f"{'='*143}")

    filled_trades = [u for u in unified if u["status"] == "FILLED"]
    partial_trades = [u for u in unified if u["status"] == "PARTIAL"]
    pending_trades = [u for u in unified if u["status"] == "PENDING"]

    t_cost = sum(u["cost_basis"] for u in unified)
    t_value = sum(u["current_value"] for u in unified)
    t_pnl = sum(u["cash_pnl"] for u in unified)
    t_pending_cap = sum(u["pending_capital"] for u in unified)
    t_committed = sum(u["total_committed"] for u in unified)
    t_filled_shares = sum(u["filled_shares"] for u in unified)
    redeemable = [u for u in unified if u["redeemable"]]

    print(f"\n  Trade Status:")
    print(f"    Fully filled:     {len(filled_trades):>3d}")
    print(f"    Partially filled: {len(partial_trades):>3d}")
    print(f"    Pending (0%):     {len(pending_trades):>3d}")
    print(f"    Total unique:     {len(unified):>3d}")

    print(f"\n  Capital:")
    print(f"    Filled (cost):    ${t_cost:>10,.2f}")
    print(f"    Pending (on book):${t_pending_cap:>10,.2f}")
    print(f"    Total committed:  ${t_committed:>10,.2f}")

    print(f"\n  Performance (filled only):")
    print(f"    Current value:    ${t_value:>10,.2f}")
    print(f"    Unrealized P&L:   ${'+'if t_pnl>=0 else ''}{t_pnl:>9,.2f} "
          f"({'+'if t_pnl>=0 else ''}{t_pnl/t_cost*100 if t_cost>0 else 0:.1f}%)")
    print(f"    Max payout:       ${t_filled_shares:>10,.2f}")
    if redeemable:
        print(f"    🎯 Redeemable:    {len(redeemable)} positions!")

    # Category breakdown
    cats = {}
    for u in unified:
        c = u["category"]
        if c not in cats:
            cats[c] = {"n": 0, "cost": 0, "value": 0, "pnl": 0,
                       "pending": 0, "shares": 0}
        cats[c]["n"] += 1
        cats[c]["cost"] += u["cost_basis"]
        cats[c]["value"] += u["current_value"]
        cats[c]["pnl"] += u["cash_pnl"]
        cats[c]["pending"] += u["pending_capital"]
        cats[c]["shares"] += u["filled_shares"]

    print(f"\n  By Category:")
    print(f"    {'Category':<10s} {'#':>3s} {'Cost':>9s} {'Value':>9s} "
          f"{'P&L':>9s} {'Pending':>9s} {'MaxPay':>9s}")
    print(f"    {'-'*55}")
    for c, s in sorted(cats.items(), key=lambda x: x[1]["cost"], reverse=True):
        pnl_s = f"{'+'if s['pnl']>=0 else ''}${s['pnl']:>7,.2f}"
        print(f"    {c:<10s} {s['n']:>3d} ${s['cost']:>8,.2f} ${s['value']:>8,.2f} "
              f"{pnl_s} ${s['pending']:>8,.2f} ${s['shares']:>8,.2f}")

    print(f"\n{'='*143}")


if __name__ == "__main__":
    main()
