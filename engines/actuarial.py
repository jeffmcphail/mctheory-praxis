"""
engines/actuarial.py — Prediction Market Actuarial Engine

Prevents inefficient capital allocation by answering:
  1. Is this position worth entering? (EV after fees, time cost)
  2. Should I hold or redeploy? (opportunity cost vs alternatives)
  3. What's the optimal portfolio allocation? (risk-adjusted returns)

Core metrics:
  - Expected Value (EV) = P(win) × payout - cost
  - Annualized Return = EV / cost / (days_to_resolve / 365)
  - Capital Efficiency = annualized_return / capital_locked
  - Opportunity Score = this position vs best alternative
  - Kelly Fraction = optimal position size given edge and odds

Compares prediction market positions against:
  - Other available Polymarket positions
  - Risk-free rate (high-yield savings, ~4.5%)
  - Your current portfolio positions

Usage:
    python -m engines.actuarial evaluate "Will BTC hit 100k?" --price 0.45 --days 90
    python -m engines.actuarial portfolio                    # Evaluate current holdings
    python -m engines.actuarial opportunities                # Best risk-adjusted opportunities
    python -m engines.actuarial opportunities --category sports --top 20
    python -m engines.actuarial compare                      # Hold vs redeploy analysis
"""
import argparse
import json
import math
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv
load_dotenv()

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
DATA_API = "https://data-api.polymarket.com"
DB_PATH = Path("data/actuarial.db")

# Constants
POLYMARKET_FEE = 0.02         # 2% on winning side
RISK_FREE_RATE = 0.045        # 4.5% annualized (high-yield savings benchmark)
MIN_VOLUME = 10000            # Minimum volume to consider
MIN_ANNUALIZED_RETURN = 0.10  # 10% annualized minimum to flag as opportunity
KELLY_FRACTION = 0.25         # Quarter Kelly (conservative)


# ═══════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            market_slug TEXT,
            question TEXT,
            side TEXT,
            entry_price REAL,
            market_price REAL,
            estimated_prob REAL,
            days_to_resolve REAL,
            ev_per_dollar REAL,
            annualized_return REAL,
            kelly_fraction REAL,
            capital_efficiency REAL,
            opportunity_score REAL,
            recommendation TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            snapshot_id TEXT,
            market_slug TEXT,
            question TEXT,
            side TEXT,
            size REAL,
            entry_price REAL,
            current_price REAL,
            cost REAL,
            current_value REAL,
            days_to_resolve REAL,
            ev_per_dollar REAL,
            annualized_return REAL,
            hold_score REAL,
            recommendation TEXT
        )
    """)

    conn.commit()
    return conn


# ═══════════════════════════════════════════════════════
# ACTUARIAL CALCULATIONS
# ═══════════════════════════════════════════════════════

def calculate_ev(price, prob_win, fee=POLYMARKET_FEE):
    """Calculate expected value per dollar invested.

    Args:
        price: cost to buy one share (0-1)
        prob_win: estimated probability of YES
        fee: winner fee (default 2%)

    Returns:
        EV per dollar invested (positive = edge)
    """
    if price <= 0 or price >= 1:
        return 0

    # If buying YES at `price`:
    # Win: receive $1 - fee = $0.98, profit = $0.98 - price
    # Lose: receive $0, loss = price
    payout_win = 1.0 - fee
    ev = prob_win * payout_win - price
    ev_per_dollar = ev / price

    return ev_per_dollar


def calculate_annualized_return(ev_per_dollar, days_to_resolve):
    """Convert per-trade EV to annualized return.

    A 10% return in 7 days is much better than 10% in 365 days.
    """
    if days_to_resolve <= 0:
        return 0

    # Compound: (1 + ev)^(365/days) - 1
    if ev_per_dollar <= -1:
        return -1

    try:
        annualized = (1 + ev_per_dollar) ** (365 / days_to_resolve) - 1
        return min(annualized, 100)  # Cap at 10000% to avoid infinity
    except (OverflowError, ValueError):
        return 100 if ev_per_dollar > 0 else -1


def calculate_kelly(prob_win, price, fee=POLYMARKET_FEE):
    """Kelly criterion for optimal position sizing.

    Returns fraction of bankroll to bet (before applying fractional Kelly).
    """
    if price <= 0 or price >= 1 or prob_win <= 0:
        return 0

    payout_win = 1.0 - fee
    # Odds offered: b = (payout_win - price) / price = net_win / stake
    b = (payout_win / price) - 1
    if b <= 0:
        return 0

    # Kelly: f = (bp - q) / b where p=prob_win, q=1-prob_win
    q = 1 - prob_win
    kelly = (b * prob_win - q) / b

    # Apply fractional Kelly for safety
    return max(0, kelly * KELLY_FRACTION)


def calculate_capital_efficiency(annualized_return, days_to_resolve):
    """Score how efficiently capital is used.

    Short-duration high-return positions score highest.
    Long-duration low-return positions score lowest.
    """
    if days_to_resolve <= 0 or annualized_return <= 0:
        return 0

    # Favor shorter durations (capital turns over faster)
    turnover_bonus = 365 / max(days_to_resolve, 1)

    # Base efficiency = annualized return × turnover potential
    efficiency = annualized_return * min(turnover_bonus, 52)  # Cap at weekly

    return efficiency


def days_until_resolution(end_date_str):
    """Calculate days until a market resolves."""
    if not end_date_str:
        return 365  # Default: assume 1 year

    try:
        # Handle various date formats
        for fmt in ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ",
                     "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"]:
            try:
                end_date = datetime.strptime(end_date_str, fmt).replace(
                    tzinfo=timezone.utc)
                delta = end_date - datetime.now(timezone.utc)
                return max(delta.total_seconds() / 86400, 0.01)
            except ValueError:
                continue
        return 365
    except Exception:
        return 365


def evaluate_position(price, prob_win, days_to_resolve, side="YES"):
    """Full actuarial evaluation of a position.

    Args:
        price: cost to buy one share
        prob_win: estimated probability of winning
        days_to_resolve: days until market resolves
        side: "YES" or "NO"

    Returns:
        dict with all metrics
    """
    # If buying NO, flip the probability
    if side == "NO":
        effective_price = 1 - price  # NO price ≈ 1 - YES price
        effective_prob = 1 - prob_win
    else:
        effective_price = price
        effective_prob = prob_win

    ev = calculate_ev(effective_price, effective_prob)
    ann_return = calculate_annualized_return(ev, days_to_resolve)
    kelly = calculate_kelly(effective_prob, effective_price)
    efficiency = calculate_capital_efficiency(ann_return, days_to_resolve)

    # Excess return over risk-free rate
    excess_return = ann_return - RISK_FREE_RATE

    # Recommendation
    if ev <= 0:
        recommendation = "AVOID"
        reason = "Negative expected value"
    elif ann_return < RISK_FREE_RATE:
        recommendation = "AVOID"
        reason = f"Below risk-free rate ({RISK_FREE_RATE:.1%})"
    elif ann_return < MIN_ANNUALIZED_RETURN:
        recommendation = "MARGINAL"
        reason = f"Below minimum threshold ({MIN_ANNUALIZED_RETURN:.0%})"
    elif days_to_resolve > 180 and ann_return < 0.20:
        recommendation = "WEAK_HOLD"
        reason = f"Long duration ({days_to_resolve:.0f}d), modest return"
    elif kelly > 0.05 and ann_return > 0.30:
        recommendation = "STRONG_BUY"
        reason = f"High EV ({ev:+.1%}), good Kelly ({kelly:.1%})"
    elif kelly > 0.02:
        recommendation = "BUY"
        reason = f"Positive edge ({ev:+.1%})"
    else:
        recommendation = "MARGINAL"
        reason = f"Thin edge ({ev:+.1%})"

    return {
        "side": side,
        "price": effective_price,
        "prob_win": effective_prob,
        "days_to_resolve": days_to_resolve,
        "ev_per_dollar": ev,
        "annualized_return": ann_return,
        "excess_return": excess_return,
        "kelly_fraction": kelly,
        "capital_efficiency": efficiency,
        "recommendation": recommendation,
        "reason": reason,
    }


# ═══════════════════════════════════════════════════════
# COMMANDS
# ═══════════════════════════════════════════════════════

def cmd_evaluate(args):
    """Evaluate a single position."""
    question = args.question
    price = args.price
    days = getattr(args, "days", 30)
    prob = getattr(args, "prob", None)
    side = getattr(args, "side", "YES")

    # If no probability given, use price as market's implied probability
    if prob is None:
        prob = price
        print(f"  (Using market price {price:.0%} as probability estimate)")

    result = evaluate_position(price, prob, days, side)

    print(f"\n{'='*70}")
    print(f"  ACTUARIAL EVALUATION")
    print(f"{'='*70}")
    print(f"  Question:           {question}")
    print(f"  Side:               {side}")
    print(f"  Entry price:        {price:.1%}")
    print(f"  Your probability:   {prob:.1%}")
    print(f"  Days to resolve:    {days:.0f}")
    print(f"{'─'*70}")
    print(f"  EV per dollar:      {result['ev_per_dollar']:+.2%}")
    print(f"  Annualized return:  {result['annualized_return']:+.1%}")
    print(f"  vs Risk-free:       {result['excess_return']:+.1%} "
          f"(benchmark: {RISK_FREE_RATE:.1%})")
    print(f"  Kelly fraction:     {result['kelly_fraction']:.2%} "
          f"(quarter Kelly)")
    print(f"  Capital efficiency: {result['capital_efficiency']:.1f}")
    print(f"{'─'*70}")

    icon = {"STRONG_BUY": "🟢", "BUY": "🟢", "MARGINAL": "🟡",
            "WEAK_HOLD": "🟡", "AVOID": "🔴"}.get(result["recommendation"], "  ")
    print(f"  Recommendation:     {icon} {result['recommendation']}")
    print(f"  Reason:             {result['reason']}")

    # Sizing example
    if result["kelly_fraction"] > 0:
        bankroll = 200  # Approximate current bankroll
        optimal_size = bankroll * result["kelly_fraction"]
        print(f"\n  Sizing (${bankroll:.0f} bankroll):")
        print(f"    Kelly optimal:    ${optimal_size:.2f}")
        print(f"    Shares:           {optimal_size / price:.1f}")

    print(f"\n{'='*70}")


def cmd_portfolio(args):
    """Evaluate all current portfolio positions."""
    from web3 import Web3

    pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    if not pk:
        print("  ❌ No POLYMARKET_PRIVATE_KEY")
        return

    w3 = Web3()
    wallet = w3.eth.account.from_key(pk).address

    print(f"\n{'='*90}")
    print(f"  PORTFOLIO ACTUARIAL ANALYSIS")
    print(f"{'='*90}")
    print(f"  Fetching positions...")

    # Get positions from Data API
    try:
        r = requests.get(f"{DATA_API}/positions",
                         params={"user": wallet}, timeout=10)
        positions = r.json()
    except Exception as e:
        print(f"  ❌ Failed to fetch positions: {e}")
        return

    if not positions:
        print(f"  No positions found.")
        return

    conn = init_db()
    snapshot_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    now = datetime.now(timezone.utc).isoformat()

    results = []

    for p in positions:
        title = p.get("title", p.get("market", {}).get("question", "?"))
        outcome = p.get("outcome", "?")
        size = float(p.get("size", 0) or 0)
        avg_price = float(p.get("avgPrice", p.get("averagePrice", 0)) or 0)
        cur_price = float(p.get("curPrice", p.get("currentPrice", 0)) or 0)

        if size <= 0 or avg_price <= 0:
            continue

        cost = size * avg_price
        value = size * cur_price if cur_price > 0 else cost

        # Get end date
        end_date = ""
        if isinstance(p.get("market"), dict):
            end_date = p["market"].get("endDate", "")
        days = days_until_resolution(end_date)

        # Use current price as implied probability
        prob = cur_price if outcome == "Yes" else (1 - cur_price)

        # Evaluate holding this position
        side = "YES" if outcome == "Yes" else "NO"
        eval_result = evaluate_position(cur_price, prob, days, side)

        # Holding analysis: what's the EV of continuing to hold?
        # If you already own at avg_price, your cost is sunk.
        # The question is: is the REMAINING upside worth the capital locked?
        remaining_upside = (1.0 - POLYMARKET_FEE - cur_price) * size if prob > cur_price else 0
        capital_locked = value
        hold_ev = remaining_upside / capital_locked if capital_locked > 0 else 0
        hold_annualized = calculate_annualized_return(hold_ev, days)

        # Hold score: should you keep this position?
        if days <= 1:
            hold_score = 10  # Resolving soon, just hold
            hold_rec = "HOLD — resolves soon"
        elif hold_annualized < RISK_FREE_RATE and value > 10:
            hold_score = 2
            hold_rec = f"CONSIDER SELL — {hold_annualized:.0%} ann. < risk-free {RISK_FREE_RATE:.0%}"
        elif hold_annualized < 0:
            hold_score = 1
            hold_rec = f"SELL — negative EV ({hold_annualized:.0%})"
        elif hold_annualized > 0.50:
            hold_score = 9
            hold_rec = f"STRONG HOLD — {hold_annualized:.0%} annualized"
        else:
            hold_score = 5
            hold_rec = f"HOLD — {hold_annualized:.0%} annualized"

        results.append({
            "title": str(title)[:55],
            "side": outcome,
            "size": size,
            "entry": avg_price,
            "current": cur_price,
            "cost": cost,
            "value": value,
            "pnl": value - cost,
            "days": days,
            "hold_ann": hold_annualized,
            "hold_score": hold_score,
            "hold_rec": hold_rec,
            "ev": eval_result["ev_per_dollar"],
            "kelly": eval_result["kelly_fraction"],
        })

        # Store
        conn.execute("""
            INSERT INTO portfolio_snapshots
            (timestamp, snapshot_id, question, side, size, entry_price,
             current_price, cost, current_value, days_to_resolve,
             ev_per_dollar, annualized_return, hold_score, recommendation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now, snapshot_id, str(title)[:100], outcome, size,
            avg_price, cur_price, cost, value, days,
            eval_result["ev_per_dollar"], hold_annualized,
            hold_score, hold_rec,
        ))

    conn.commit()

    # Sort by hold score (worst first — candidates to sell)
    results.sort(key=lambda x: x["hold_score"])

    # Display
    total_cost = sum(r["cost"] for r in results)
    total_value = sum(r["value"] for r in results)
    total_pnl = total_value - total_cost

    print(f"\n  Positions: {len(results)} | "
          f"Cost: ${total_cost:.0f} | Value: ${total_value:.0f} | "
          f"P&L: ${total_pnl:+.0f}")

    print(f"\n  {'Market':<45s} {'Side':<4s} {'Cost':>6s} {'Value':>6s} "
          f"{'P&L':>7s} {'Days':>5s} {'Ann%':>6s} {'Score':>5s} {'Action'}")
    print(f"  {'─'*120}")

    for r in results:
        icon = "🔴" if r["hold_score"] <= 2 else ("🟡" if r["hold_score"] <= 5 else "🟢")
        ann_str = f"{r['hold_ann']:+.0%}" if abs(r['hold_ann']) < 100 else "∞"
        print(f"  {r['title']:<45s} {r['side']:<4s} "
              f"${r['cost']:>5.0f} ${r['value']:>5.0f} "
              f"${r['pnl']:>+6.0f} {r['days']:>5.0f} "
              f"{ann_str:>6s} {r['hold_score']:>5.0f} {icon} {r['hold_rec']}")

    # Capital redeployment analysis
    sell_candidates = [r for r in results if r["hold_score"] <= 2]
    if sell_candidates:
        freed_capital = sum(r["value"] for r in sell_candidates)
        print(f"\n  💡 REDEPLOYMENT OPPORTUNITY:")
        print(f"     Selling {len(sell_candidates)} underperforming positions "
              f"would free ${freed_capital:.0f}")
        print(f"     At risk-free rate ({RISK_FREE_RATE:.1%}): "
              f"${freed_capital * RISK_FREE_RATE:.2f}/year guaranteed")
        print(f"     Better deployed in high-efficiency short-duration markets")

    conn.close()
    print(f"\n{'='*90}")


def cmd_opportunities(args):
    """Scan for the best risk-adjusted opportunities on Polymarket."""
    top_n = getattr(args, "top", 30)
    category = getattr(args, "category", None)

    print(f"\n{'='*90}")
    print(f"  OPPORTUNITY SCANNER — Best Risk-Adjusted Bets")
    print(f"  Scanning top {top_n} markets | Min annualized: {MIN_ANNUALIZED_RETURN:.0%}")
    print(f"{'='*90}")

    # Fetch markets
    all_markets = []
    offset = 0
    while len(all_markets) < top_n * 3:
        try:
            r = requests.get(f"{GAMMA_API}/markets", params={
                "closed": "false", "active": "true",
                "limit": 100, "offset": offset,
            }, timeout=15)
            batch = r.json()
            if not batch:
                break
            all_markets.extend(batch)
            offset += 100
            if len(batch) < 100:
                break
        except Exception:
            break

    # Filter
    candidates = []
    for m in all_markets:
        vol = float(m.get("volume", 0) or 0)
        if vol < MIN_VOLUME:
            continue

        token_ids = json.loads(m.get("clobTokenIds", "[]"))
        if len(token_ids) < 1:
            continue

        end_date = m.get("endDate", "")
        days = days_until_resolution(end_date)

        # Skip already resolved or resolving today
        if days < 0.5:
            continue

        # Get current price
        try:
            r = requests.get(f"{CLOB_API}/midpoint",
                             params={"token_id": token_ids[0]}, timeout=5)
            data = r.json()
            mid = float(data.get("mid", 0.5) if isinstance(data, dict) else data)
        except Exception:
            continue

        # Skip extreme prices
        if mid < 0.05 or mid > 0.95:
            continue

        candidates.append({
            "slug": m.get("slug", ""),
            "question": m.get("question", ""),
            "volume": vol,
            "price": mid,
            "end_date": end_date,
            "days": days,
        })

        time.sleep(0.05)  # Rate limit

    print(f"  Found {len(candidates)} candidates")

    # Evaluate each: both YES and NO side
    opportunities = []

    for c in candidates:
        # Use market price as probability (efficient market assumption)
        # The "edge" comes from our AI ensemble or other analysis
        # Here we just rank by capital efficiency assuming fair pricing
        price = c["price"]
        days = c["days"]

        # YES side
        yes_eval = evaluate_position(price, price, days, "YES")
        # NO side
        no_eval = evaluate_position(price, price, days, "NO")

        # For opportunity scanning with market-implied probabilities,
        # EV is ~0 (market is fairly priced). Instead, rank by
        # what return you'd get IF you have even a small edge.

        # Assume 5% edge (you think the true prob is 5pp different)
        edge = 0.05
        yes_with_edge = evaluate_position(price, price + edge, days, "YES")
        no_with_edge = evaluate_position(1 - price, (1 - price) + edge, days, "NO")

        best = yes_with_edge if yes_with_edge["annualized_return"] > no_with_edge["annualized_return"] else no_with_edge
        best_side = "YES" if best == yes_with_edge else "NO"

        opportunities.append({
            "slug": c["slug"],
            "question": c["question"][:50],
            "volume": c["volume"],
            "price": price,
            "days": days,
            "side": best_side,
            "ann_return_5pct_edge": best["annualized_return"],
            "kelly_5pct_edge": best["kelly_fraction"],
            "efficiency": best["capital_efficiency"],
            "recommendation": best["recommendation"],
        })

    # Sort by capital efficiency
    opportunities.sort(key=lambda x: -x["efficiency"])

    # Display top opportunities
    print(f"\n  Top opportunities (assuming 5% edge over market):\n")
    print(f"  {'Market':<45s} {'Price':>5s} {'Side':<4s} {'Days':>5s} "
          f"{'Ann%':>7s} {'Kelly':>6s} {'Eff':>6s} {'Vol':>10s}")
    print(f"  {'─'*95}")

    for o in opportunities[:top_n]:
        ann_str = f"{o['ann_return_5pct_edge']:+.0%}" if abs(o['ann_return_5pct_edge']) < 100 else ">100x"
        print(f"  {o['question']:<45s} {o['price']:>4.0%} {o['side']:<4s} "
              f"{o['days']:>5.0f} {ann_str:>7s} "
              f"{o['kelly_5pct_edge']:>5.1%} {o['efficiency']:>6.1f} "
              f"${o['volume']:>9,.0f}")

    print(f"\n  Key insight: Short-duration markets with moderate prices")
    print(f"  (30-70¢) offer the best capital efficiency. A 5% edge on a")
    print(f"  7-day market annualizes to {calculate_annualized_return(0.05/0.50, 7):.0%} vs "
          f"{calculate_annualized_return(0.05/0.50, 180):.0%} for a 180-day market.")

    print(f"\n{'='*90}")


def cmd_compare(args):
    """Compare holding current positions vs redeploying capital."""
    from web3 import Web3

    pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    if not pk:
        print("  ❌ No POLYMARKET_PRIVATE_KEY")
        return

    w3 = Web3()
    wallet = w3.eth.account.from_key(pk).address

    print(f"\n{'='*90}")
    print(f"  HOLD vs REDEPLOY ANALYSIS")
    print(f"{'='*90}")

    # Get current positions
    try:
        r = requests.get(f"{DATA_API}/positions",
                         params={"user": wallet}, timeout=10)
        positions = r.json()
    except Exception:
        print(f"  ❌ Failed to fetch positions")
        return

    if not positions:
        print(f"  No positions.")
        return

    # Evaluate each position's hold value
    hold_analysis = []

    for p in positions:
        title = p.get("title", p.get("market", {}).get("question", "?"))
        outcome = p.get("outcome", "?")
        size = float(p.get("size", 0) or 0)
        avg_price = float(p.get("avgPrice", p.get("averagePrice", 0)) or 0)
        cur_price = float(p.get("curPrice", p.get("currentPrice", 0)) or 0)

        if size <= 0 or avg_price <= 0:
            continue

        cost = size * avg_price
        value = size * cur_price if cur_price > 0 else cost

        end_date = ""
        if isinstance(p.get("market"), dict):
            end_date = p["market"].get("endDate", "")
        days = days_until_resolution(end_date)

        # Value of holding: expected additional profit from current price to resolution
        if outcome == "Yes":
            # If we hold YES and it resolves YES: get $0.98/share - already paid avg_price
            # Probability of winning ≈ current price
            hold_ev = cur_price * (0.98 - cur_price) * size  # Expected remaining gain
        else:
            hold_ev = (1 - cur_price) * (0.98 - (1 - cur_price)) * size

        hold_ann = calculate_annualized_return(
            hold_ev / value if value > 0 else 0, days)

        # Value of selling: get `value` cash now, deploy at risk-free rate
        sell_value = value
        risk_free_gain = sell_value * RISK_FREE_RATE * (days / 365)

        # Net comparison
        hold_advantage = hold_ev - risk_free_gain

        hold_analysis.append({
            "title": str(title)[:45],
            "side": outcome,
            "value": value,
            "days": days,
            "hold_ev": hold_ev,
            "hold_ann": hold_ann,
            "sell_rf_gain": risk_free_gain,
            "advantage": hold_advantage,
            "action": "HOLD" if hold_advantage > 0 else "REDEPLOY",
        })

    # Sort by advantage (worst holds first)
    hold_analysis.sort(key=lambda x: x["advantage"])

    print(f"\n  {'Market':<40s} {'Side':<4s} {'Value':>7s} {'Days':>5s} "
          f"{'HoldEV':>8s} {'RF$':>7s} {'Adv':>8s} {'Action'}")
    print(f"  {'─'*100}")

    for h in hold_analysis:
        icon = "🟢" if h["action"] == "HOLD" else "🔴"
        print(f"  {h['title']:<40s} {h['side']:<4s} "
              f"${h['value']:>6.0f} {h['days']:>5.0f} "
              f"${h['hold_ev']:>7.2f} ${h['sell_rf_gain']:>6.2f} "
              f"${h['advantage']:>+7.2f} {icon} {h['action']}")

    # Summary
    redeploy = [h for h in hold_analysis if h["action"] == "REDEPLOY"]
    if redeploy:
        total_freed = sum(h["value"] for h in redeploy)
        total_saved = sum(-h["advantage"] for h in redeploy)
        print(f"\n  💡 RECOMMENDATION: Sell {len(redeploy)} positions to free ${total_freed:.0f}")
        print(f"     Expected improvement: ${total_saved:.2f} over hold period")
        print(f"     Freed capital can be deployed in higher-efficiency markets")
    else:
        print(f"\n  ✅ All positions are worth holding (beat risk-free rate)")

    print(f"\n{'='*90}")


def main():
    parser = argparse.ArgumentParser(description="Prediction Market Actuarial Engine")
    subs = parser.add_subparsers(dest="command")

    p_eval = subs.add_parser("evaluate", help="Evaluate a single position")
    p_eval.add_argument("question", type=str)
    p_eval.add_argument("--price", type=float, required=True)
    p_eval.add_argument("--prob", type=float, default=None,
                        help="Your probability estimate (default: use market price)")
    p_eval.add_argument("--days", type=float, default=30)
    p_eval.add_argument("--side", default="YES", choices=["YES", "NO"])

    subs.add_parser("portfolio", help="Evaluate current holdings")

    p_opp = subs.add_parser("opportunities", help="Best opportunities")
    p_opp.add_argument("--top", type=int, default=30)
    p_opp.add_argument("--category", default=None)

    subs.add_parser("compare", help="Hold vs redeploy analysis")

    args = parser.parse_args()

    if args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "portfolio":
        cmd_portfolio(args)
    elif args.command == "opportunities":
        cmd_opportunities(args)
    elif args.command == "compare":
        cmd_compare(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
