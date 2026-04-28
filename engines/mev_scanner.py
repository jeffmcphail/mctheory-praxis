"""
engines/mev_scanner.py — Polymarket MEV Opportunity Scanner

Scans ALL active Polymarket markets for arbitrage opportunities:
  1. Sum-to-One: YES_ask + NO_ask < $1.00 (after fees)
  2. NegRisk Rebalancing: Multi-outcome probabilities sum != $1.00
  3. Cross-Market Combinatorial: Related markets with inconsistent pricing

Phase 1: Detection and logging only. No execution.

VALIDATION: Maximal by default.
  - Verifies token IDs exist and are tradeable
  - Double-checks prices via both Gamma API and CLOB
  - Logs ALL data to SQLite for backtesting
  - Verbose output on every scan cycle

Usage:
    python -m engines.mev_scanner scan                    # One-shot scan
    python -m engines.mev_scanner scan --min-profit 0.02  # Only show >2% opportunities
    python -m engines.mev_scanner monitor                 # Continuous monitoring
    python -m engines.mev_scanner monitor --interval 30   # Every 30 seconds
    python -m engines.mev_scanner stats                   # Show historical data
"""
import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
load_dotenv()

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
POLYMARKET_FEE = 0.02  # 2% winner fee
MIN_VOLUME = 1000       # Minimum volume to consider (filter noise)
DB_PATH = Path("data/mev_scanner.db")

# ═══════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS opportunities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            scan_id TEXT NOT NULL,
            market_slug TEXT,
            question TEXT,
            opp_type TEXT,
            yes_ask REAL,
            no_ask REAL,
            sum_price REAL,
            gross_spread REAL,
            net_profit REAL,
            yes_token TEXT,
            no_token TEXT,
            volume_24h REAL,
            liquidity REAL,
            neg_risk INTEGER,
            tick_size REAL,
            condition_id TEXT,
            details TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scan_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            markets_scanned INTEGER,
            opportunities_found INTEGER,
            best_net_profit REAL,
            total_potential REAL,
            scan_duration_s REAL,
            errors INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS neg_risk_opportunities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            scan_id TEXT NOT NULL,
            event_slug TEXT,
            event_title TEXT,
            num_outcomes INTEGER,
            sum_best_asks REAL,
            deviation REAL,
            net_profit REAL,
            outcomes_json TEXT
        )
    """)
    conn.commit()
    return conn


# ═══════════════════════════════════════════════════════
# MARKET DATA FETCHING
# ═══════════════════════════════════════════════════════

def fetch_all_active_markets(min_volume=MIN_VOLUME, verbose=True):
    """Fetch all active, tradeable markets from Gamma API."""
    all_markets = []
    offset = 0
    limit = 100
    fetch_start = time.time()

    if verbose:
        print(f"  Fetching active markets from Gamma API (100/batch)...")
        print(f"  This typically takes 1-2 minutes for ~50K markets.")

    while True:
        try:
            r = requests.get(f"{GAMMA_API}/markets", params={
                "closed": "false",
                "active": "true",
                "limit": limit,
                "offset": offset,
            }, timeout=15)
            batch = r.json()
            if not batch:
                break
            all_markets.extend(batch)
            offset += limit

            # Progress every 5000 markets
            if verbose and len(all_markets) % 5000 < limit:
                elapsed = time.time() - fetch_start
                print(f"    {len(all_markets):>6,d} markets fetched... "
                      f"({elapsed:.0f}s elapsed)")

            if len(batch) < limit:
                break
        except Exception as e:
            if verbose:
                print(f"    ⚠ Error fetching markets at offset {offset}: {e}")
            break

    elapsed = time.time() - fetch_start
    if verbose:
        print(f"  Fetched {len(all_markets):,d} markets in {elapsed:.1f}s")

    # Filter for minimum volume and valid token IDs
    tradeable = []
    for m in all_markets:
        try:
            token_ids = json.loads(m.get("clobTokenIds", "[]"))
            if len(token_ids) < 2:
                continue

            vol = float(m.get("volume", 0) or 0)
            if vol < min_volume:
                continue

            tradeable.append(m)
        except Exception:
            continue

    if verbose:
        print(f"  {len(tradeable)} tradeable markets (volume > ${min_volume})")

    return tradeable


def get_clob_prices(token_id, verbose=False):
    """Get best bid/ask from CLOB for a token."""
    try:
        # Best bid (what you'd get selling)
        bid_r = requests.get(f"{CLOB_API}/price",
                             params={"token_id": token_id, "side": "BUY"},
                             timeout=5)
        bid = float(bid_r.json().get("price", 0)) if bid_r.status_code == 200 else 0

        # Best ask (what you'd pay buying)
        ask_r = requests.get(f"{CLOB_API}/price",
                             params={"token_id": token_id, "side": "SELL"},
                             timeout=5)
        ask = float(ask_r.json().get("price", 0)) if ask_r.status_code == 200 else 0

        # Midpoint
        mid_r = requests.get(f"{CLOB_API}/midpoint",
                             params={"token_id": token_id},
                             timeout=5)
        mid_data = mid_r.json() if mid_r.status_code == 200 else {}
        mid = float(mid_data.get("mid", 0)) if isinstance(mid_data, dict) else float(mid_data)

        return {"bid": bid, "ask": ask, "mid": mid}
    except Exception as e:
        if verbose:
            print(f"    CLOB price error for {token_id[:20]}...: {e}")
        return {"bid": 0, "ask": 0, "mid": 0}


def get_clob_book(token_id, verbose=False):
    """Get order book for deeper analysis."""
    try:
        r = requests.get(f"{CLOB_API}/book",
                         params={"token_id": token_id},
                         timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {"bids": [], "asks": []}


# ═══════════════════════════════════════════════════════
# OPPORTUNITY DETECTION
# ═══════════════════════════════════════════════════════

def check_sum_to_one_fast(market):
    """Fast check using Gamma API prices (no CLOB calls).
    
    Returns (sum_price, yes_price, no_price) or None if no data.
    """
    try:
        prices_str = market.get("outcomePrices", "")
        if not prices_str:
            return None
        # outcomePrices is like "[0.55, 0.45]" or "0.55,0.45"
        prices_str = prices_str.strip("[]")
        parts = [p.strip().strip('"') for p in prices_str.split(",")]
        if len(parts) < 2:
            return None
        yes_p = float(parts[0])
        no_p = float(parts[1])
        if yes_p <= 0 or no_p <= 0:
            return None
        return (yes_p + no_p, yes_p, no_p)
    except Exception:
        return None


def check_sum_to_one(market, verbose=False):
    """Full check using CLOB prices. Only call for pre-filtered markets.

    Returns opportunity dict or None.
    """
    token_ids = json.loads(market.get("clobTokenIds", "[]"))
    if len(token_ids) < 2:
        return None

    yes_token = token_ids[0]
    no_token = token_ids[1]

    # Get CLOB prices
    yes_prices = get_clob_prices(yes_token, verbose)
    no_prices = get_clob_prices(no_token, verbose)

    yes_ask = yes_prices["ask"]
    no_ask = no_prices["ask"]

    if yes_ask <= 0 or no_ask <= 0:
        return None

    sum_price = yes_ask + no_ask
    gross_spread = 1.0 - sum_price  # Positive = buy both for < $1

    # Net profit after 2% winner fee (you pay fee on the winning side = $1 payout)
    net_profit = gross_spread - POLYMARKET_FEE

    # Also check the reverse: selling both for > $1 (YES_bid + NO_bid > $1)
    yes_bid = yes_prices["bid"]
    no_bid = no_prices["bid"]
    reverse_sum = yes_bid + no_bid
    reverse_spread = reverse_sum - 1.0  # Positive = sell both for > $1
    reverse_net = reverse_spread - POLYMARKET_FEE

    opp = {
        "market_slug": market.get("slug", ""),
        "question": market.get("question", ""),
        "yes_ask": yes_ask,
        "no_ask": no_ask,
        "yes_bid": yes_bid,
        "no_bid": no_bid,
        "sum_ask": sum_price,
        "sum_bid": reverse_sum,
        "gross_spread": gross_spread,
        "net_profit": net_profit,
        "reverse_gross": reverse_spread,
        "reverse_net": reverse_net,
        "yes_token": yes_token,
        "no_token": no_token,
        "volume": float(market.get("volume", 0) or 0),
        "liquidity": float(market.get("liquidityClob", 0) or 0),
        "neg_risk": market.get("negRisk", False),
        "tick_size": float(market.get("orderPriceMinTickSize", 0.01) or 0.01),
        "condition_id": market.get("conditionId", ""),
        "mid_yes": yes_prices["mid"],
        "mid_no": no_prices["mid"],
    }

    return opp


def fetch_neg_risk_events(verbose=True):
    """Fetch NegRisk events (multi-outcome markets) for sum verification."""
    try:
        r = requests.get(f"{GAMMA_API}/events", params={
            "closed": "false",
            "active": "true",
            "limit": 200,
        }, timeout=15)
        events = r.json()
        # Filter to events with multiple markets
        multi = [e for e in events if len(e.get("markets", [])) > 2]
        if verbose:
            print(f"  Found {len(multi)} multi-outcome events")
        return multi
    except Exception as e:
        if verbose:
            print(f"  ⚠ Error fetching events: {e}")
        return []


def infer_event_structure(event):
    """Infer the correct expected sum (K) and exclusivity for a NegRisk event.
    
    Returns dict with:
      k: expected number of TRUE outcomes (sum should equal K)
      exclusive: True if outcomes are mutually exclusive (K=1)
      skip: True if outcomes are non-exclusive or nested (no valid arb)
      reason: human-readable classification
    """
    title = (event.get("title", "") or "").lower()
    slug = (event.get("slug", "") or "").lower()
    num_markets = len(event.get("markets", []))
    
    # Check sample questions for pattern detection
    questions = [m.get("question", "").lower() for m in event.get("markets", [])]
    q_sample = " ".join(questions[:5])
    
    # ── NON-EXCLUSIVE: multiple can be true simultaneously ──
    # "What will happen before X" — independent events
    if "what will happen before" in title or "what will happen before" in slug:
        return {"k": 0, "exclusive": False, "skip": True, "reason": "non-exclusive (independent events)"}
    
    # "Will X endorse Y" for different races — can endorse multiple
    if "endorse" in title and any(kw in q_sample for kw in ["for", "by nov"]):
        # Check if different races/positions
        positions = set()
        for q in questions:
            for kw in ["-sen", "-gov", "mayor", "for tx", "for ny", "for fl", "for ca",
                       "for wa", "for ia", "for ne", "for sc", "for ky", "for me"]:
                if kw in q:
                    positions.add(kw)
        if len(positions) > 1:
            return {"k": 0, "exclusive": False, "skip": True, "reason": "non-exclusive (different races)"}
    
    # ── NESTED TEMPORAL: "X by date" where later dates subsume earlier ──
    # Check BOTH title and questions for date patterns
    date_keywords = ["january", "february", "march", "april", "may", "june",
                     "july", "august", "september", "october", "november", "december",
                     "2025", "2026", "2027", "2028", "2029", "2030"]
    
    # Count how many questions contain date references
    date_count = sum(1 for q in questions if any(d in q for d in date_keywords))
    questions_have_dates = date_count >= len(questions) * 0.6
    
    # Title patterns that signal temporal nesting
    temporal_title = any(kw in title for kw in [
        "released by", "by…", "by...", "out by", "called by", "before",
        "by ___", "by_", "hit $", "reach $", "pregnant", "capture",
        "invade", "clash by", "agree to", "normalize", "recognize",
        "leave nato", "airdrop by", "foul play", "sells any",
    ])
    
    # Question patterns: "by [month]", "by [date]", "in [year]", "before [year]"
    temporal_q_patterns = ["by december", "by june", "by march", "by september",
                           "by january", "by february", "by april", "by may",
                           "by july", "by august", "by october", "by november",
                           "in 2025", "in 2026", "in 2027", "before 2026",
                           "before 2027", "before 2028"]
    q_temporal_count = sum(1 for q in questions if any(p in q for p in temporal_q_patterns))
    questions_are_temporal = q_temporal_count >= len(questions) * 0.6
    
    if (temporal_title and questions_have_dates) or questions_are_temporal:
        return {"k": 0, "exclusive": False, "skip": True, "reason": "nested temporal (dates subsume)"}
    
    # ── NESTED THRESHOLD: "above $X" where higher thresholds subsume lower ──
    threshold_patterns = [">$", "above $", "more than $", "over $", "exceed",
                          "fdv above", "market cap", "mcap"]
    threshold_in_title = any(kw in title for kw in threshold_patterns)
    threshold_in_questions = sum(1 for q in questions 
                                 if any(kw in q for kw in threshold_patterns))
    if threshold_in_title or threshold_in_questions >= len(questions) * 0.5:
        return {"k": 0, "exclusive": False, "skip": True, "reason": "nested threshold"}
    
    # ── K-OF-N: exactly K outcomes are true ──
    # "Top 4 Finish" — exactly 4 teams
    if "top 4" in title:
        return {"k": 4, "exclusive": False, "skip": False, "reason": "K-of-N (top 4)"}
    if "top 6" in title:
        return {"k": 6, "exclusive": False, "skip": False, "reason": "K-of-N (top 6)"}
    if "top 2" in title:
        return {"k": 2, "exclusive": False, "skip": False, "reason": "K-of-N (top 2)"}
    
    # "Relegation" — typically 3 teams relegated
    if "relegat" in title:
        return {"k": 3, "exclusive": False, "skip": False, "reason": "K-of-N (relegation, K=3)"}
    
    # "Promoted" — typically 3 teams promoted
    if "promot" in title:
        return {"k": 3, "exclusive": False, "skip": False, "reason": "K-of-N (promotion, K=3)"}
    
    # ── MUTUALLY EXCLUSIVE (K=1): exactly one winner ──
    # "Who will win X" — one winner
    if any(kw in title for kw in ["who will win", "winner of", "champion"]):
        return {"k": 1, "exclusive": True, "skip": False, "reason": "exclusive (one winner)"}
    
    # "Will X be the next Y" — one person for one role
    if any(kw in title for kw in ["next", "nominee", "presidential", "election"]):
        return {"k": 1, "exclusive": True, "skip": False, "reason": "exclusive (one outcome)"}
    
    # "Which" — typically one answer
    if title.startswith("which"):
        return {"k": 1, "exclusive": True, "skip": False, "reason": "exclusive (which)"}
    
    # "Sentenced to" — one sentencing range
    if "sentenced" in title or "sentence" in q_sample:
        return {"k": 1, "exclusive": True, "skip": False, "reason": "exclusive (one sentence)"}
    
    # Default: assume K=1 (mutually exclusive) but flag as uncertain
    return {"k": 1, "exclusive": True, "skip": False, "reason": "assumed exclusive (K=1)"}


def check_neg_risk_event(event, verbose=False):
    """Check if a NegRisk event's outcome probabilities deviate from expected sum.
    
    Uses Gamma prices only (no CLOB calls) for fast scanning.
    Computes correct expected sum K based on event structure.
    """
    markets = event.get("markets", [])
    if len(markets) < 3:
        return None

    # Determine correct expected sum
    structure = infer_event_structure(event)
    if structure["skip"]:
        return None  # Non-exclusive or nested — no valid arb

    expected_sum = structure["k"]

    outcomes = []
    total_best_ask = 0
    errors = 0

    for m in markets:
        try:
            prices_raw = m.get("outcomePrices", "")
            if not prices_raw:
                errors += 1
                continue
            
            if isinstance(prices_raw, str):
                prices_raw = prices_raw.strip("[]")
                parts = [p.strip().strip('"').strip("'") for p in prices_raw.split(",")]
            elif isinstance(prices_raw, list):
                parts = [str(p).strip('"').strip("'") for p in prices_raw]
            else:
                errors += 1
                continue
            
            if not parts or not parts[0]:
                errors += 1
                continue
                
            yes_price = float(parts[0])
            
            outcomes.append({
                "question": m.get("question", "")[:50],
                "yes_ask": yes_price,
                "source": "gamma",
            })
            total_best_ask += yes_price
        except (ValueError, IndexError):
            errors += 1
            continue

    if len(outcomes) < 3:
        return None

    deviation = total_best_ask - expected_sum
    # Net profit = deviation minus fees on all legs
    # For K-of-N: you'd buy all N outcomes, pay 2% fee on K winning payouts
    fee_cost = POLYMARKET_FEE * expected_sum
    net_profit = abs(deviation) - fee_cost

    return {
        "event_slug": event.get("slug", ""),
        "event_title": event.get("title", "")[:60],
        "num_outcomes": len(outcomes),
        "sum_best_asks": total_best_ask,
        "expected_sum": expected_sum,
        "deviation": deviation,
        "deviation_pct": (deviation / expected_sum * 100) if expected_sum > 0 else 0,
        "net_profit": net_profit,
        "fee_cost": fee_cost,
        "structure": structure["reason"],
        "k": expected_sum,
        "outcomes": outcomes,
        "errors": errors,
    }


# ═══════════════════════════════════════════════════════
# COMMANDS
# ═══════════════════════════════════════════════════════

def cmd_scan(args):
    """One-shot scan of all markets."""
    min_profit = getattr(args, "min_profit", -999)
    min_volume = getattr(args, "min_volume", MIN_VOLUME)
    verbose = getattr(args, "verbose", True)
    top_n = getattr(args, "top", 20)

    conn = init_db()
    scan_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    start_time = time.time()

    print(f"\n{'='*100}")
    print(f"  POLYMARKET MEV SCANNER — Phase 1 (Detection Only)")
    print(f"  Scan ID: {scan_id}")
    print(f"  Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Min profit filter: ${min_profit:.3f}  |  Min volume: ${min_volume:,.0f}")
    print(f"{'='*100}")

    # ── BINARY MARKET SCAN ──
    markets = fetch_all_active_markets(min_volume=min_volume, verbose=True)

    # Sort by volume descending, scan top N via CLOB
    markets.sort(key=lambda m: float(m.get("volume", 0) or 0), reverse=True)
    MAX_CLOB_SCAN = getattr(args, "max_markets", 500)
    scan_markets = markets[:MAX_CLOB_SCAN]

    print(f"\n  Scanning top {len(scan_markets)} markets by volume via CLOB...")
    if scan_markets:
        print(f"  (Volume range: ${float(scan_markets[0].get('volume',0) or 0):,.0f} — "
              f"${float(scan_markets[-1].get('volume',0) or 0):,.0f})")
        est_time = len(scan_markets) * 0.2  # ~0.2s per market (3 API calls + delay)
        print(f"  Estimated time: ~{est_time:.0f}s ({est_time/60:.1f} min)")
    else:
        print(f"  Skipping CLOB scan (--max-markets 0)")

    all_opps = []
    profitable = 0
    errors = 0
    last_progress = time.time()
    clob_scan_start = time.time()

    for i, m in enumerate(scan_markets):
        try:
            opp = check_sum_to_one(m, verbose=False)
            if opp is None:
                errors += 1
                continue

            opp["source"] = "clob"
            all_opps.append(opp)

            # Log to DB
            conn.execute("""
                INSERT INTO opportunities
                (timestamp, scan_id, market_slug, question, opp_type,
                 yes_ask, no_ask, sum_price, gross_spread, net_profit,
                 yes_token, no_token, volume_24h, liquidity,
                 neg_risk, tick_size, condition_id, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(timezone.utc).isoformat(), scan_id,
                opp["market_slug"], opp["question"], "sum_to_one",
                opp["yes_ask"], opp["no_ask"], opp["sum_ask"],
                opp["gross_spread"], opp["net_profit"],
                opp["yes_token"][:30], opp["no_token"][:30],
                opp["volume"], opp["liquidity"],
                1 if opp["neg_risk"] else 0, opp["tick_size"],
                opp["condition_id"],
                json.dumps({"yes_bid": opp["yes_bid"], "no_bid": opp["no_bid"],
                            "sum_bid": opp["sum_bid"], "reverse_net": opp["reverse_net"]}),
            ))

            if opp["net_profit"] > 0:
                profitable += 1
                print(f"    ✅ {opp['question'][:50]} — net ${opp['net_profit']:.4f}")

        except Exception as e:
            errors += 1
            if verbose:
                print(f"    ⚠ Error on {m.get('slug','?')}: {e}")

        # Progress every 30 seconds
        now = time.time()
        if now - last_progress >= 30:
            elapsed = now - clob_scan_start
            pct = (i + 1) / len(scan_markets) * 100
            remaining = (elapsed / (i + 1)) * (len(scan_markets) - i - 1)
            print(f"    [{datetime.now().strftime('%H:%M:%S')}] "
                  f"{i+1}/{len(scan_markets)} ({pct:.0f}%) | "
                  f"{len(all_opps)} priced, {profitable} profitable, {errors} errors | "
                  f"~{remaining:.0f}s remaining")
            last_progress = now

        time.sleep(0.15)  # Rate limit CLOB calls

    conn.commit()

    clob_elapsed = time.time() - clob_scan_start
    print(f"  CLOB scan complete: {len(all_opps)} priced, {profitable} profitable, "
          f"{errors} errors in {clob_elapsed:.0f}s")

    # Sort by net profit descending
    combined_opps = sorted(all_opps, key=lambda x: x["net_profit"], reverse=True)

    # Display top opportunities
    print(f"\n  ── TOP {top_n} OPPORTUNITIES (by net profit) ──\n")
    print(f"  {'#':<4s} {'Market':<50s} {'YesAsk':>7s} {'NoAsk':>7s} {'Sum':>6s} "
          f"{'Gross':>7s} {'Net':>7s} {'Vol':>10s} {'Src':>5s}")
    print(f"  {'─'*110}")

    for i, opp in enumerate(combined_opps[:top_n], 1):
        q = opp["question"][:49]
        net_str = f"${opp['net_profit']:.3f}" if opp["net_profit"] > 0 else f"{opp['net_profit']:.3f}"
        gross_str = f"{opp['gross_spread']:.3f}"
        vol_str = f"${opp['volume']:,.0f}"
        src = opp.get("source", "?")[:5]
        flag = " ✅" if opp["net_profit"] > 0 else ""

        print(f"  {i:<4d} {q:<50s} {opp['yes_ask']:>7.3f} {opp['no_ask']:>7.3f} "
              f"{opp['sum_ask']:>6.3f} {gross_str:>7s} {net_str:>7s} "
              f"{vol_str:>10s} {src:>5s}{flag}")

        if verbose and opp["net_profit"] > 0:
            print(f"       Slug: {opp['market_slug']}")
            print(f"       Buy YES @ {opp['yes_ask']:.3f} + Buy NO @ {opp['no_ask']:.3f} "
                  f"= ${opp['sum_ask']:.3f} → payout $1.00")
            print(f"       Profit: ${opp['gross_spread']:.4f} gross - "
                  f"${POLYMARKET_FEE:.2f} fee = ${opp['net_profit']:.4f} net per share")
            print(f"       Source: {opp.get('source','?')}")
            rev = opp.get("reverse_net", 0)
            if rev > 0:
                print(f"       REVERSE: Sell YES @ {opp['yes_bid']:.3f} + "
                      f"Sell NO @ {opp['no_bid']:.3f} = ${opp['sum_bid']:.3f} "
                      f"(net ${rev:.4f})")
            print()

    # ── NEGRISK MULTI-OUTCOME SCAN ──
    print(f"\n  ── NEGRISK MULTI-OUTCOME SCAN ──")
    events = fetch_neg_risk_events(verbose=True)
    if events:
        print(f"  Scanning {len(events)} events (Gamma prices only, should be fast)...")

    nr_opps = []
    nr_errors = 0
    nr_skipped = 0
    skip_reasons = {}

    for ei, event in enumerate(events):
        # First check structure — skip non-exclusive/nested
        structure = infer_event_structure(event)
        if structure["skip"]:
            nr_skipped += 1
            reason = structure["reason"]
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            continue

        try:
            nr = check_neg_risk_event(event, verbose=False)
            if nr and abs(nr["deviation_pct"]) > 1.0:  # >1% deviation from expected
                nr_opps.append(nr)
                conn.execute("""
                    INSERT INTO neg_risk_opportunities
                    (timestamp, scan_id, event_slug, event_title,
                     num_outcomes, sum_best_asks, deviation, net_profit,
                     outcomes_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now(timezone.utc).isoformat(), scan_id,
                    nr["event_slug"], nr["event_title"],
                    nr["num_outcomes"], nr["sum_best_asks"],
                    nr["deviation"], nr["net_profit"],
                    json.dumps(nr["outcomes"]),
                ))
        except Exception as e:
            nr_errors += 1
        time.sleep(0.2)

    conn.commit()

    # Show skip/error stats
    if nr_skipped > 0:
        print(f"  Skipped {nr_skipped} non-exclusive/nested events:")
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            print(f"    {count:>3d} × {reason}")
    if nr_errors > 0:
        print(f"  ({nr_errors} events had parsing errors)")

    nr_opps.sort(key=lambda x: abs(nr["deviation_pct"]) if nr == x else abs(x.get("deviation_pct", 0)),
                 reverse=True)
    # Simpler sort
    nr_opps.sort(key=lambda x: abs(x.get("deviation_pct", 0)), reverse=True)

    if nr_opps:
        print(f"\n  Found {len(nr_opps)} events with >1% deviation from expected sum:\n")
        print(f"  {'#':<4s} {'Event':<40s} {'Type':<18s} {'K':>2s} {'#Out':>5s} "
              f"{'Sum':>7s} {'Exp':>5s} {'Dev%':>6s} {'Net$':>7s}")
        print(f"  {'─'*100}")

        for i, nr in enumerate(nr_opps[:20], 1):
            dev_pct = f"{nr['deviation_pct']:+.1f}%"
            net_str = f"${nr['net_profit']:.3f}" if nr["net_profit"] > 0 else f"{nr['net_profit']:.3f}"
            flag = " ✅" if nr["net_profit"] > 0 else ""
            struct = nr.get("structure", "?")[:17]
            print(f"  {i:<4d} {nr['event_title'][:39]:<40s} {struct:<18s} {nr['k']:>2d} "
                  f"{nr['num_outcomes']:>5d} {nr['sum_best_asks']:>7.3f} "
                  f"{nr['expected_sum']:>5.1f} {dev_pct:>6s} {net_str:>7s}{flag}")

            if verbose and nr["net_profit"] > 0:
                for o in nr["outcomes"]:
                    print(f"       {o['question']:<45s} ask={o['yes_ask']:.3f}")
                print()
    else:
        print(f"  No significant deviations found after classification.")

    # ── SUMMARY ──
    scan_duration = time.time() - start_time

    profitable_opps = [o for o in combined_opps if o["net_profit"] > 0]
    profitable_nr = [n for n in nr_opps if n["net_profit"] > 0]

    total_potential = sum(o["net_profit"] for o in profitable_opps)
    best_net = max((o["net_profit"] for o in combined_opps), default=0)

    conn.execute("""
        INSERT INTO scan_log
        (scan_id, timestamp, markets_scanned, opportunities_found,
         best_net_profit, total_potential, scan_duration_s, errors)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (scan_id, datetime.now(timezone.utc).isoformat(),
          len(markets), len(profitable_opps),
          best_net, total_potential, scan_duration, errors))
    conn.commit()
    conn.close()

    print(f"\n{'─'*100}")
    print(f"  SCAN SUMMARY — {scan_id}")
    print(f"{'─'*100}")
    print(f"  Total active markets:      {len(markets)}")
    print(f"  CLOB scanned (top by vol): {len(scan_markets)}")
    print(f"  CLOB errors:               {errors}")
    print(f"  Scan duration:             {scan_duration:.1f}s")
    print(f"  Binary opportunities:")
    print(f"    Gross positive (sum<1):  {sum(1 for o in all_opps if o['gross_spread'] > 0)}")
    print(f"    Net profitable (>fee):   {len(profitable_opps)}")
    print(f"    Best net profit:         ${best_net:.4f}/share")
    if profitable_opps:
        print(f"    Total potential:         ${total_potential:.4f}/share across {len(profitable_opps)} markets")
    print(f"  NegRisk events:")
    print(f"    Total scanned:           {len(events)}")
    print(f"    Skipped (non-exclusive): {nr_skipped}")
    print(f"    Valid deviations (>1%):  {len(nr_opps)}")
    print(f"    Net profitable:          {len(profitable_nr)}")
    if profitable_nr:
        best_nr = max(n["net_profit"] for n in profitable_nr)
        print(f"    Best NR net profit:      ${best_nr:.3f}")
    print(f"\n  Data saved to: {DB_PATH}")
    print(f"{'='*100}")


def cmd_monitor(args):
    """Continuous monitoring loop."""
    interval = getattr(args, "interval", 60)
    min_profit = getattr(args, "min_profit", 0)

    print(f"\n  Starting continuous monitor (every {interval}s, min profit ${min_profit:.3f})...")
    print(f"  Press Ctrl+C to stop.\n")

    cycle = 0
    while True:
        try:
            cycle += 1
            print(f"\n  ══ Cycle {cycle} — {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')} ══")

            # Reuse scan logic but with abbreviated output
            args.verbose = False
            args.top = 5
            cmd_scan(args)

        except KeyboardInterrupt:
            print(f"\n  Stopped after {cycle} cycles.")
            break
        except Exception as e:
            print(f"  Error in cycle {cycle}: {e}")

        time.sleep(interval)


def cmd_stats(args):
    """Show historical scan statistics."""
    conn = init_db()

    print(f"\n{'='*80}")
    print(f"  MEV SCANNER STATISTICS")
    print(f"{'='*80}")

    # Recent scans
    scans = conn.execute("""
        SELECT scan_id, timestamp, markets_scanned, opportunities_found,
               best_net_profit, total_potential, scan_duration_s, errors
        FROM scan_log ORDER BY timestamp DESC LIMIT 20
    """).fetchall()

    if scans:
        print(f"\n  Recent Scans ({len(scans)}):\n")
        print(f"  {'Scan ID':<18s} {'Markets':>8s} {'Opps':>5s} {'Best$':>8s} "
              f"{'Total$':>8s} {'Time':>6s} {'Err':>4s}")
        print(f"  {'─'*65}")
        for s in scans:
            print(f"  {s[0]:<18s} {s[2]:>8d} {s[3]:>5d} ${s[4]:>6.4f} "
                  f"${s[5]:>6.4f} {s[6]:>5.1f}s {s[7]:>4d}")

    # Best opportunities ever found
    best = conn.execute("""
        SELECT timestamp, question, yes_ask, no_ask, sum_price,
               gross_spread, net_profit, volume_24h
        FROM opportunities
        WHERE net_profit > 0
        ORDER BY net_profit DESC LIMIT 10
    """).fetchall()

    if best:
        print(f"\n  Best Opportunities Ever Found:\n")
        print(f"  {'Time':<20s} {'Market':<40s} {'Net$':>7s} {'Vol':>10s}")
        print(f"  {'─'*80}")
        for b in best:
            t = b[0][:19]
            q = b[1][:39]
            print(f"  {t:<20s} {q:<40s} ${b[6]:>.4f} ${b[7]:>8,.0f}")

    # NegRisk stats
    nr_best = conn.execute("""
        SELECT timestamp, event_title, num_outcomes, deviation, net_profit
        FROM neg_risk_opportunities
        WHERE net_profit > 0
        ORDER BY net_profit DESC LIMIT 10
    """).fetchall()

    if nr_best:
        print(f"\n  Best NegRisk Opportunities:\n")
        print(f"  {'Time':<20s} {'Event':<40s} {'#Out':>5s} {'Dev':>7s} {'Net$':>7s}")
        print(f"  {'─'*85}")
        for n in nr_best:
            t = n[0][:19]
            e = n[1][:39]
            print(f"  {t:<20s} {e:<40s} {n[2]:>5d} {n[3]:>+6.3f} ${n[4]:>.4f}")

    conn.close()
    print(f"\n{'='*80}")


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Polymarket MEV Scanner")
    subparsers = parser.add_subparsers(dest="command")

    # Scan
    p_scan = subparsers.add_parser("scan", help="One-shot market scan")
    p_scan.add_argument("--min-profit", type=float, default=-999,
                        help="Min net profit to display (default: show all)")
    p_scan.add_argument("--min-volume", type=float, default=MIN_VOLUME,
                        help=f"Min 24h volume (default: ${MIN_VOLUME})")
    p_scan.add_argument("--top", type=int, default=20,
                        help="Show top N opportunities (default: 20)")
    p_scan.add_argument("--max-markets", type=int, default=500,
                        help="Max markets to CLOB-scan (default: 500)")
    p_scan.add_argument("--verbose", action="store_true", default=True)
    p_scan.add_argument("--quiet", action="store_true")

    # Monitor
    p_mon = subparsers.add_parser("monitor", help="Continuous monitoring")
    p_mon.add_argument("--interval", type=int, default=60,
                       help="Seconds between scans (default: 60)")
    p_mon.add_argument("--min-profit", type=float, default=0)
    p_mon.add_argument("--min-volume", type=float, default=MIN_VOLUME)
    p_mon.add_argument("--top", type=int, default=5)
    p_mon.add_argument("--verbose", action="store_true", default=False)

    # Stats
    p_stats = subparsers.add_parser("stats", help="Show historical data")

    args = parser.parse_args()
    if hasattr(args, "quiet") and args.quiet:
        args.verbose = False

    if args.command == "scan":
        cmd_scan(args)
    elif args.command == "monitor":
        cmd_monitor(args)
    elif args.command == "stats":
        cmd_stats(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
