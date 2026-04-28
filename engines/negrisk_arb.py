"""
engines/negrisk_arb.py -- Manual NegRisk Arbitrage Executor

When the sum of all outcome prices in a NegRisk event is less than K
(usually 1.0), you can:
  1. Buy all N outcomes on CLOB (off-chain, sequential)
  2. Merge all outcomes on-chain via NegRisk Adapter -> receive K USDC.e per set
  3. Pocket the difference minus fees

This is NOT a flash loan -- it uses real capital. The arb is NOT atomic
because CLOB orders are off-chain. Prices can move between buys.

Risk management:
  - Pre-check: verify sum still below threshold before starting
  - Position tracking: resume interrupted buys
  - Abort threshold: stop if prices move against us mid-execution
  - Max capital limit per trade

Usage:
    python -m engines.negrisk_arb scan                         # Find opportunities
    python -m engines.negrisk_arb analyze "michigan-governor"  # Deep dive on one event
    python -m engines.negrisk_arb execute "michigan-governor"  # Paper trade
    python -m engines.negrisk_arb execute "michigan-governor" --live  # Real execution
    python -m engines.negrisk_arb merge                        # Merge held positions
    python -m engines.negrisk_arb status                       # Check in-progress arbs
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
DATA_API = "https://data-api.polymarket.com"
DB_PATH = Path("data/negrisk_arb.db")

# NegRisk Adapter on Polygon
NEGRISK_ADAPTER = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"
CTF_CONTRACT = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
USDC_E = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"

POLYMARKET_FEE = 0.02
DEFAULT_MAX_CAPITAL = 200  # Max USD per arb
ABORT_SLIPPAGE = 0.03      # Abort if sum moves 3% against us during execution


# ===================================================================
# DATABASE
# ===================================================================

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS arb_executions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            event_slug TEXT NOT NULL,
            event_title TEXT,
            n_outcomes INTEGER,
            expected_sum REAL,
            actual_sum_at_start REAL,
            target_profit_pct REAL,
            max_capital REAL,
            status TEXT DEFAULT 'PENDING',
            outcomes_bought INTEGER DEFAULT 0,
            total_cost REAL DEFAULT 0,
            merge_tx TEXT,
            merge_payout REAL,
            actual_profit REAL,
            notes TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS arb_legs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            execution_id INTEGER NOT NULL,
            outcome_index INTEGER NOT NULL,
            market_slug TEXT,
            question TEXT,
            token_id TEXT,
            target_price REAL,
            fill_price REAL,
            size REAL,
            cost REAL,
            order_id TEXT,
            status TEXT DEFAULT 'PENDING',
            timestamp TEXT,
            FOREIGN KEY (execution_id) REFERENCES arb_executions(id)
        )
    """)

    conn.commit()
    return conn


# ===================================================================
# MARKET DATA
# ===================================================================

def fetch_event(slug):
    """Fetch a NegRisk event by slug or title substring."""
    slug_lower = slug.lower()

    # Try exact slug match first
    try:
        r = requests.get(f"{GAMMA_API}/events", params={
            "slug": slug, "closed": "false",
        }, timeout=15)
        events = r.json()
        if events:
            return events[0]
    except Exception:
        pass

    # Fall back: fetch all and search by slug or title
    try:
        r = requests.get(f"{GAMMA_API}/events", params={
            "closed": "false", "limit": 200,
        }, timeout=15)
        all_events = r.json()
        for e in all_events:
            if e.get("slug", "") == slug:
                return e
            # Fuzzy title match
            title = (e.get("title", "") or "").lower()
            if slug_lower in title or title in slug_lower:
                return e
        # Even looser: check if all words in slug appear in title
        words = slug_lower.replace("-", " ").replace("_", " ").split()
        for e in all_events:
            title = (e.get("title", "") or "").lower()
            if all(w in title for w in words):
                return e
    except Exception:
        pass

    return None


def fetch_event_prices(event):
    """Fetch current CLOB prices for all outcomes in an event."""
    markets = event.get("markets", [])
    outcomes = []

    for i, m in enumerate(markets):
        token_ids = json.loads(m.get("clobTokenIds", "[]"))
        if not token_ids:
            continue

        # Get midpoint
        try:
            r = requests.get(f"{CLOB_API}/midpoint",
                             params={"token_id": token_ids[0]}, timeout=5)
            data = r.json()
            mid = float(data.get("mid", 0) if isinstance(data, dict) else data)
        except Exception:
            mid = 0

        # Get best ask (what we'd pay to buy)
        try:
            r = requests.get(f"{CLOB_API}/book",
                             params={"token_id": token_ids[0]}, timeout=5)
            book = r.json()
            asks = book.get("asks", [])
            best_ask = float(asks[0].get("price", mid)) if asks else mid
            ask_size = float(asks[0].get("size", 0)) if asks else 0
        except Exception:
            best_ask = mid
            ask_size = 0

        outcomes.append({
            "index": i,
            "question": m.get("question", "")[:60],
            "slug": m.get("slug", ""),
            "token_id_yes": token_ids[0] if token_ids else "",
            "token_id_no": token_ids[1] if len(token_ids) > 1 else "",
            "midpoint": mid,
            "best_ask": best_ask,
            "ask_size": ask_size,
            "condition_id": m.get("conditionId", ""),
        })

        time.sleep(0.05)  # Rate limit

    return outcomes


def infer_k(event):
    """Infer expected sum K for a NegRisk event."""
    title = (event.get("title", "") or "").lower()
    if "top 4" in title or "top four" in title:
        return 4
    if "top 3" in title or "top three" in title:
        return 3
    if "relegate" in title or "relegation" in title:
        return 3
    return 1


# ===================================================================
# COMMANDS
# ===================================================================

def cmd_scan(args):
    """Find profitable NegRisk arb opportunities."""
    min_profit = getattr(args, "min_profit", 2.0)

    print(f"\n{'='*90}")
    print(f"  NEGRISK ARB SCANNER")
    print(f"  Min profit: {min_profit}% after fees | Max capital: ${DEFAULT_MAX_CAPITAL}")
    print(f"{'='*90}")

    # Fetch all NegRisk events
    try:
        r = requests.get(f"{GAMMA_API}/events", params={
            "closed": "false", "limit": 200,
        }, timeout=15)
        events = r.json()
    except Exception:
        print(f"  Failed to fetch events")
        return

    neg_risk = [e for e in events if e.get("negRisk")]
    total_outcomes = sum(len(e.get("markets", [])) for e in neg_risk)
    print(f"  NegRisk events: {len(neg_risk)} ({total_outcomes} total outcomes)")
    print(f"  Estimated time: Pass 1 ~{total_outcomes * 0.03:.0f}s, Pass 2 depends on candidates")
    print(f"")
    print(f"  ---- PASS 1: Fast midpoint pre-screen ----")

    opportunities = []
    phantom_count = 0

    # -- PASS 1: Fast midpoint pre-screen --
    candidates = []
    pass1_start = time.time()
    midpoint_calls = 0

    for ei, event in enumerate(neg_risk):
        title = event.get("title", "")
        slug = event.get("slug", "")
        markets = event.get("markets", [])

        if len(markets) < 2:
            continue

        k = infer_k(event)

        prices = []
        for m in markets:
            token_ids = json.loads(m.get("clobTokenIds", "[]"))
            if not token_ids:
                break
            try:
                r = requests.get(f"{CLOB_API}/midpoint",
                                 params={"token_id": token_ids[0]}, timeout=5)
                data = r.json()
                mid = float(data.get("mid", 0) if isinstance(data, dict) else data)
                prices.append(mid)
                midpoint_calls += 1
            except Exception:
                break
            time.sleep(0.02)

        if len(prices) != len(markets):
            continue

        total_mid = sum(prices)
        deviation_pct = (k - total_mid) / k * 100

        elapsed = time.time() - pass1_start
        print(f"    [{ei+1:>3d}/{len(neg_risk)}] {title[:45]:<45s} "
              f"N={len(markets):>3d} sum={total_mid:.3f} "
              f"dev={deviation_pct:>+5.1f}% "
              f"{'** CANDIDATE **' if deviation_pct > min_profit else ''}"
              f"  ({elapsed:.0f}s, {midpoint_calls} API calls)")

        if deviation_pct > min_profit:
            candidates.append({
                "event": event, "title": title, "slug": slug,
                "k": k, "n": len(markets), "mid_sum": total_mid,
                "mid_dev_pct": deviation_pct,
            })

    pass1_elapsed = time.time() - pass1_start
    print(f"")
    print(f"  Pass 1 complete: {len(neg_risk)} events, {midpoint_calls} API calls, "
          f"{pass1_elapsed:.0f}s")
    print(f"  Candidates with midpoint deviation > {min_profit}%: {len(candidates)}")

    if candidates:
        print(f"")
        print(f"  Candidate summary:")
        for i, c in enumerate(candidates):
            print(f"    {i+1}. {c['title'][:50]} | N={c['n']} | "
                  f"mid_sum={c['mid_sum']:.3f} | dev={c['mid_dev_pct']:+.1f}%")

    if not candidates:
        print(f"\n  No events with midpoint deviation > {min_profit}%.")
        return

    # -- PASS 2: Verify with order books (only candidates) --
    print(f"")
    print(f"  ---- PASS 2: Order book liquidity verification ----")
    total_p2_outcomes = sum(c["n"] for c in candidates)
    print(f"  Checking {len(candidates)} candidates ({total_p2_outcomes} outcomes)")
    print(f"  Estimated time: ~{total_p2_outcomes * 0.1:.0f}s")
    print(f"")
    pass2_start = time.time()
    book_calls = 0

    for ci, cand in enumerate(candidates):
        event = cand["event"]
        markets = event.get("markets", [])
        k = cand["k"]

        print(f"    [{ci+1}/{len(candidates)}] {cand['title'][:50]} "
              f"(N={cand['n']}, checking {len(markets)} order books...)")

        ask_prices = []
        mid_prices = []
        all_liquid = True
        illiquid_outcome = None

        for mi, m in enumerate(markets):
            token_ids = json.loads(m.get("clobTokenIds", "[]"))
            if not token_ids:
                all_liquid = False
                illiquid_outcome = m.get("question", f"outcome {mi}")[:40]
                break
            try:
                r = requests.get(f"{CLOB_API}/book",
                                 params={"token_id": token_ids[0]}, timeout=5)
                book = r.json()
                asks = book.get("asks", [])
                book_calls += 1

                if not asks or float(asks[0].get("size", 0)) == 0:
                    all_liquid = False
                    illiquid_outcome = m.get("question", f"outcome {mi}")[:40]
                    print(f"      outcome {mi+1}/{len(markets)}: NO ASKS -- "
                          f"{illiquid_outcome}")
                    break

                best_ask = float(asks[0]["price"])
                ask_size = float(asks[0].get("size", 0))
                ask_prices.append(best_ask)

                bids = book.get("bids", [])
                if bids:
                    mid_prices.append((float(bids[0]["price"]) + best_ask) / 2)
                else:
                    mid_prices.append(best_ask)

            except Exception as e:
                all_liquid = False
                illiquid_outcome = f"API error on outcome {mi}: {e}"
                break
            time.sleep(0.05)

        p2_elapsed = time.time() - pass2_start

        if not all_liquid:
            phantom_count += 1
            print(f"      --> PHANTOM (illiquid: {illiquid_outcome}) "
                  f"[{p2_elapsed:.0f}s, {book_calls} book calls]")
            continue

        total_ask = sum(ask_prices)
        total_mid = sum(mid_prices)
        deviation = k - total_ask
        deviation_pct = deviation / k * 100
        net_profit_pct = deviation_pct - (POLYMARKET_FEE * 100)

        if net_profit_pct >= min_profit:
            sets_buyable = DEFAULT_MAX_CAPITAL / total_ask if total_ask > 0 else 0
            dollar_profit = sets_buyable * (k * (1 - POLYMARKET_FEE) - total_ask)

            opportunities.append({
                "title": cand["title"][:50], "slug": cand["slug"],
                "n": cand["n"], "k": k,
                "sum_mid": total_mid, "sum_ask": total_ask, "sum": total_ask,
                "deviation": deviation, "dev_pct": deviation_pct,
                "net_pct": net_profit_pct, "dollar_profit": dollar_profit,
            })
            print(f"      --> REAL OPPORTUNITY | mid={cand['mid_sum']:.3f} "
                  f"ask={total_ask:.3f} | net={net_profit_pct:+.1f}% "
                  f"${dollar_profit:.0f} "
                  f"[{p2_elapsed:.0f}s, {book_calls} book calls]")
        else:
            print(f"      --> UNPROFITABLE at ask prices | "
                  f"mid={cand['mid_sum']:.3f} ask={total_ask:.3f} | "
                  f"net={net_profit_pct:+.1f}% "
                  f"[{p2_elapsed:.0f}s, {book_calls} book calls]")

    pass2_elapsed = time.time() - pass2_start
    total_elapsed = time.time() - pass1_start
    print(f"")
    print(f"  Pass 2 complete: {book_calls} book calls, {pass2_elapsed:.0f}s")
    print(f"  Total scan time: {total_elapsed:.0f}s")
    print(f"  {phantom_count} phantom (illiquid) filtered out")

    if not opportunities:
        print(f"\n  No opportunities above {min_profit}% net profit with full liquidity.")
        print(f"  Many NegRisk events have phantom mispricings (illiquid outcomes).")
        print(f"  These are filtered out because you can't actually buy them.")
        return

    opportunities.sort(key=lambda x: -x["net_pct"])

    print(f"\n  All opportunities have verified ask-side liquidity on every outcome.\n")
    print(f"  {'Event':<40s} {'N':>3s} {'Mid':>6s} {'Ask':>6s} "
          f"{'Net%':>6s} {'$Profit':>8s}")
    print(f"  {'-'*75}")

    for o in opportunities:
        print(f"  {o['title']:<40s} {o['n']:>3d} {o['sum_mid']:>5.3f} "
              f"{o['sum_ask']:>5.3f} {o['net_pct']:>+5.1f}% "
              f"${o['dollar_profit']:>7.0f}")
        print(f"    slug: {o['slug']}")

    print(f"\n  Use: python -m engines.negrisk_arb analyze \"SLUG\" for details")
    print(f"{'='*90}")


def cmd_analyze(args):
    """Deep analysis of a specific NegRisk event for arb."""
    slug = args.slug

    print(f"\n{'='*90}")
    print(f"  NEGRISK ARB ANALYSIS")
    print(f"{'='*90}")

    event = fetch_event(slug)
    if not event:
        print(f"  Event not found: {slug}")
        print(f"  Run: python -m engines.negrisk_arb scan to find events")
        return

    title = event.get("title", "")
    markets = event.get("markets", [])
    k = infer_k(event)

    print(f"  Event: {title}")
    print(f"  Slug:  {slug}")
    print(f"  Outcomes: {len(markets)} | K={k}")
    print(f"\n  Fetching live prices...")

    outcomes = fetch_event_prices(event)

    if not outcomes:
        print(f"  Failed to get prices")
        return

    # Display all outcomes
    total_mid = sum(o["midpoint"] for o in outcomes)
    total_ask = sum(o["best_ask"] for o in outcomes)

    print(f"\n  {'#':>3s} {'Outcome':<50s} {'Mid':>6s} {'Ask':>6s} {'AskSz':>7s}")
    print(f"  {'-'*80}")

    for o in outcomes:
        print(f"  {o['index']+1:>3d} {o['question']:<50s} "
              f"{o['midpoint']:>5.3f} {o['best_ask']:>5.3f} "
              f"{o['ask_size']:>7.0f}")

    print(f"  {'-'*80}")
    print(f"  {'SUM':<54s} {total_mid:>5.3f} {total_ask:>5.3f}")

    # Profitability at midpoint
    mid_dev = k - total_mid
    mid_net = mid_dev - (k * POLYMARKET_FEE)

    # Profitability at ask (what we'd actually pay)
    ask_dev = k - total_ask
    ask_net = ask_dev - (k * POLYMARKET_FEE)

    print(f"\n  -- PROFITABILITY --")
    print(f"  At midpoint:  sum={total_mid:.4f} | gap={mid_dev:+.4f} | "
          f"net after 2% fee={mid_net:+.4f} ({mid_net/total_mid*100:+.1f}%)")
    print(f"  At best ask:  sum={total_ask:.4f} | gap={ask_dev:+.4f} | "
          f"net after 2% fee={ask_net:+.4f} ({ask_net/total_ask*100:+.1f}%)")

    # Sizing
    if ask_net > 0:
        max_cap = DEFAULT_MAX_CAPITAL
        sets = max_cap / total_ask
        dollar_profit = sets * ask_net

        print(f"\n  -- SIZING (${max_cap:.0f} capital) --")
        print(f"  Sets buyable:  {sets:.1f}")
        print(f"  Total cost:    ${sets * total_ask:.2f}")
        print(f"  Merge payout:  ${sets * k * (1-POLYMARKET_FEE):.2f}")
        print(f"  Net profit:    ${dollar_profit:.2f} ({ask_net/total_ask*100:.1f}%)")

        # Min order book depth needed
        min_depth = min(o["ask_size"] for o in outcomes if o["ask_size"] > 0) if any(o["ask_size"] > 0 for o in outcomes) else 0
        print(f"\n  Thinnest ask book: {min_depth:.0f} shares")
        if min_depth > 0 and sets > min_depth:
            print(f"  WARNING: Need {sets:.0f} shares but thinnest book has {min_depth:.0f}")
            print(f"  Will need to sweep multiple price levels = higher cost")
    else:
        print(f"\n  NOT PROFITABLE at current ask prices.")
        print(f"  Need sum to drop below {k * (1 - POLYMARKET_FEE):.4f}")

    print(f"\n  To execute: python -m engines.negrisk_arb execute \"{slug}\"")
    print(f"{'='*90}")


def cmd_execute(args):
    """Execute a NegRisk arb trade."""
    slug = args.slug
    live = getattr(args, "live", False)
    max_capital = getattr(args, "max_capital", DEFAULT_MAX_CAPITAL)

    mode = "LIVE" if live else "PAPER"

    print(f"\n{'='*90}")
    print(f"  NEGRISK ARB EXECUTOR -- {mode} MODE")
    print(f"  Max capital: ${max_capital:.0f}")
    print(f"{'='*90}")

    if live:
        pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
        if not pk:
            print(f"  No POLYMARKET_PRIVATE_KEY")
            return

    # Fetch event
    event = fetch_event(slug)
    if not event:
        print(f"  Event not found: {slug}")
        return

    title = event.get("title", "")
    markets = event.get("markets", [])
    k = infer_k(event)

    print(f"  Event: {title}")
    print(f"  Outcomes: {len(markets)} | K={k}")

    # Get fresh prices
    outcomes = fetch_event_prices(event)
    total_ask = sum(o["best_ask"] for o in outcomes)
    ask_net = (k - total_ask) - (k * POLYMARKET_FEE)

    if ask_net <= 0:
        print(f"\n  NOT PROFITABLE. Sum={total_ask:.4f}, need < {k*(1-POLYMARKET_FEE):.4f}")
        print(f"  Aborting.")
        return

    sets = max_capital / total_ask
    dollar_profit = sets * ask_net

    print(f"\n  Pre-trade check:")
    print(f"    Sum of asks:   {total_ask:.4f}")
    print(f"    Net per set:   {ask_net:+.4f} ({ask_net/total_ask*100:+.1f}%)")
    print(f"    Sets to buy:   {sets:.1f}")
    print(f"    Expected cost: ${sets * total_ask:.2f}")
    print(f"    Expected profit: ${dollar_profit:.2f}")

    # Record execution
    conn = init_db()
    now = datetime.now(timezone.utc).isoformat()

    cursor = conn.execute("""
        INSERT INTO arb_executions
        (timestamp, event_slug, event_title, n_outcomes, expected_sum,
         actual_sum_at_start, target_profit_pct, max_capital, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'IN_PROGRESS')
    """, (now, slug, title[:100], len(markets), k, total_ask,
          ask_net/total_ask*100, max_capital))
    exec_id = cursor.lastrowid
    conn.commit()

    if not live:
        print(f"\n  PAPER MODE -- simulating buys:")
        total_cost = 0

        for o in outcomes:
            size = sets  # Buy `sets` shares of each outcome
            cost = size * o["best_ask"]
            total_cost += cost

            print(f"    [{o['index']+1}/{len(outcomes)}] "
                  f"{o['question'][:40]} | "
                  f"BUY {size:.1f} @ {o['best_ask']:.3f} = ${cost:.2f}")

            conn.execute("""
                INSERT INTO arb_legs
                (execution_id, outcome_index, market_slug, question,
                 token_id, target_price, fill_price, size, cost,
                 status, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'FILLED', ?)
            """, (exec_id, o["index"], o["slug"], o["question"][:100],
                  o["token_id_yes"], o["best_ask"], o["best_ask"],
                  size, cost, now))

        merge_payout = sets * k * (1 - POLYMARKET_FEE)
        profit = merge_payout - total_cost

        conn.execute("""
            UPDATE arb_executions SET
                status='COMPLETE_PAPER', outcomes_bought=?,
                total_cost=?, merge_payout=?, actual_profit=?
            WHERE id=?
        """, (len(outcomes), total_cost, merge_payout, profit, exec_id))
        conn.commit()

        print(f"\n  PAPER RESULTS:")
        print(f"    Total cost:    ${total_cost:.2f}")
        print(f"    Merge payout:  ${merge_payout:.2f}")
        print(f"    Net profit:    ${profit:.2f} ({profit/total_cost*100:.1f}%)")
        print(f"\n  To execute for real: add --live flag")

    else:
        # LIVE EXECUTION
        print(f"\n  LIVE EXECUTION -- buying {len(outcomes)} outcomes...")

        try:
            from py_clob_client.client import ClobClient
        except ImportError:
            print(f"  pip install py-clob-client")
            return

        pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
        client = ClobClient(CLOB_API, key=pk, chain_id=137)

        total_cost = 0
        bought = 0

        for o in outcomes:
            # Re-check price before each buy (prices may have moved)
            try:
                r = requests.get(f"{CLOB_API}/book",
                                 params={"token_id": o["token_id_yes"]}, timeout=5)
                book = r.json()
                asks = book.get("asks", [])
                current_ask = float(asks[0]["price"]) if asks else o["best_ask"]
            except Exception:
                current_ask = o["best_ask"]

            # Abort check: has the sum moved too much?
            # (Simple version: check if this outcome's price jumped)
            if current_ask > o["best_ask"] * (1 + ABORT_SLIPPAGE):
                print(f"    ABORT: {o['question'][:40]} ask jumped "
                      f"{o['best_ask']:.3f} -> {current_ask:.3f}")
                conn.execute(
                    "UPDATE arb_executions SET status='ABORTED', notes=? WHERE id=?",
                    (f"Price slippage on outcome {o['index']}", exec_id))
                conn.commit()
                print(f"\n  Execution aborted. {bought}/{len(outcomes)} legs filled.")
                print(f"  You hold partial positions -- sell manually or wait for merge.")
                return

            size = sets
            cost = size * current_ask

            print(f"    [{o['index']+1}/{len(outcomes)}] "
                  f"{o['question'][:40]} | "
                  f"BUY {size:.1f} @ {current_ask:.3f} = ${cost:.2f}")

            try:
                order = client.create_and_post_order(
                    token_id=o["token_id_yes"],
                    side="BUY",
                    size=size,
                    price=round(current_ask + 0.01, 2),  # Pay 1c above ask for fill
                )

                order_id = order.get("orderID", order.get("id", ""))
                print(f"      Order: {order_id}")

                conn.execute("""
                    INSERT INTO arb_legs
                    (execution_id, outcome_index, market_slug, question,
                     token_id, target_price, fill_price, size, cost,
                     order_id, status, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'SUBMITTED', ?)
                """, (exec_id, o["index"], o["slug"], o["question"][:100],
                      o["token_id_yes"], o["best_ask"], current_ask,
                      size, cost, order_id, now))
                conn.commit()

                total_cost += cost
                bought += 1

            except Exception as e:
                print(f"      FAILED: {e}")
                conn.execute("""
                    INSERT INTO arb_legs
                    (execution_id, outcome_index, market_slug, question,
                     token_id, target_price, size, status, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'FAILED', ?)
                """, (exec_id, o["index"], o["slug"], o["question"][:100],
                      o["token_id_yes"], o["best_ask"], size, now))
                conn.commit()

            time.sleep(0.5)  # Rate limit between orders

        # Update execution record
        conn.execute("""
            UPDATE arb_executions SET
                outcomes_bought=?, total_cost=?, status=?
            WHERE id=?
        """, (bought, total_cost,
              "ALL_BOUGHT" if bought == len(outcomes) else "PARTIAL",
              exec_id))
        conn.commit()

        print(f"\n  Bought {bought}/{len(outcomes)} outcomes")
        print(f"  Total cost: ${total_cost:.2f}")

        if bought == len(outcomes):
            print(f"\n  All outcomes acquired! Ready to merge.")
            print(f"  Run: python -m engines.negrisk_arb merge --execution {exec_id}")
        else:
            print(f"\n  PARTIAL FILL -- {len(outcomes) - bought} outcomes missing")
            print(f"  Options:")
            print(f"    1. Retry missing legs: python -m engines.negrisk_arb retry --execution {exec_id}")
            print(f"    2. Sell what you have: manually sell positions")

    conn.close()
    print(f"{'='*90}")


def cmd_merge(args):
    """Merge held outcome tokens back to USDC.e on-chain."""
    exec_id = getattr(args, "execution", None)

    print(f"\n{'='*90}")
    print(f"  NEGRISK MERGE")
    print(f"{'='*90}")

    pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    if not pk:
        print(f"  No POLYMARKET_PRIVATE_KEY")
        return

    from web3 import Web3

    rpc = os.getenv("POLYGON_RPC_URL", "https://polygon-bor-rpc.publicnode.com")
    w3 = Web3(Web3.HTTPProvider(rpc))
    acct = w3.eth.account.from_key(pk)

    print(f"  Wallet: {acct.address}")
    print(f"  RPC: {rpc}")

    # NegRisk adapter mergePositions ABI (minimal)
    MERGE_ABI = [{
        "inputs": [
            {"name": "conditionId", "type": "bytes32"},
            {"name": "amount", "type": "uint256"},
        ],
        "name": "mergePositions",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    }]

    adapter = w3.eth.contract(
        address=w3.to_checksum_address(NEGRISK_ADAPTER),
        abi=MERGE_ABI
    )

    if exec_id:
        conn = init_db()
        execution = conn.execute(
            "SELECT * FROM arb_executions WHERE id=?", (exec_id,)
        ).fetchone()

        if not execution:
            print(f"  Execution {exec_id} not found")
            return

        print(f"  Event: {execution[3]}")  # event_title
        print(f"  Status: {execution[9]}")  # status

        # Get the condition ID from the event
        event = fetch_event(execution[2])  # event_slug
        if not event:
            print(f"  Could not fetch event")
            return

        condition_id = event.get("conditionId", event.get("condition_id", ""))
        if not condition_id:
            # Try to get from first market
            markets = event.get("markets", [])
            if markets:
                condition_id = markets[0].get("conditionId", "")

        if not condition_id:
            print(f"  No conditionId found for this event")
            print(f"  You may need to find it manually on Polygonscan")
            return

        print(f"  Condition ID: {condition_id}")

        # Determine merge amount (smallest position across all outcomes)
        legs = conn.execute(
            "SELECT * FROM arb_legs WHERE execution_id=? AND status='FILLED'",
            (exec_id,)
        ).fetchall()

        if not legs:
            print(f"  No filled legs found")
            return

        min_size = min(l[7] for l in legs)  # size column
        amount_raw = int(min_size * 1_000_000)  # USDC has 6 decimals

        print(f"  Merge amount: {min_size:.1f} sets ({amount_raw} raw)")

        # Build and send merge tx
        print(f"\n  Building merge transaction...")

        try:
            nonce = w3.eth.get_transaction_count(acct.address)
            gas_price = w3.eth.gas_price

            tx = adapter.functions.mergePositions(
                bytes.fromhex(condition_id.replace("0x", "")),
                amount_raw,
            ).build_transaction({
                "from": acct.address,
                "nonce": nonce,
                "gas": 500000,
                "gasPrice": gas_price,
                "chainId": 137,
            })

            signed = acct.sign_transaction(tx)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            print(f"  Tx: {tx_hash.hex()}")

            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            print(f"  Status: {'SUCCESS' if receipt.status == 1 else 'FAILED'}")
            print(f"  Gas used: {receipt.gasUsed}")
            print(f"  https://polygonscan.com/tx/{tx_hash.hex()}")

            if receipt.status == 1:
                conn.execute("""
                    UPDATE arb_executions SET
                        status='MERGED', merge_tx=?
                    WHERE id=?
                """, (tx_hash.hex(), exec_id))
                conn.commit()

        except Exception as e:
            print(f"  Merge failed: {e}")
            print(f"  The conditionId or amount may be wrong.")
            print(f"  Check the event on Polygonscan for correct parameters.")

    else:
        print(f"  Specify --execution ID")
        print(f"  Run: python -m engines.negrisk_arb status to see executions")

    print(f"{'='*90}")


def cmd_status(args):
    """Check status of all arb executions."""
    conn = init_db()

    executions = conn.execute(
        "SELECT * FROM arb_executions ORDER BY id DESC LIMIT 20"
    ).fetchall()

    print(f"\n{'='*90}")
    print(f"  ARB EXECUTION STATUS")
    print(f"{'='*90}")

    if not executions:
        print(f"  No executions yet.")
    else:
        print(f"\n  {'ID':>4s} {'Time':<20s} {'Event':<35s} {'N':>3s} "
              f"{'Cost':>8s} {'Profit':>8s} {'Status'}")
        print(f"  {'-'*90}")

        for e in executions:
            print(f"  {e[0]:>4d} {e[1][:19]:<20s} {str(e[3])[:34]:<35s} "
                  f"{e[4]:>3d} ${e[11] or 0:>7.0f} "
                  f"${e[13] or 0:>7.0f} {e[9]}")

    conn.close()
    print(f"{'='*90}")


def main():
    parser = argparse.ArgumentParser(description="NegRisk Arbitrage Executor")
    subs = parser.add_subparsers(dest="command")

    p_scan = subs.add_parser("scan", help="Find opportunities")
    p_scan.add_argument("--min-profit", type=float, default=2.0)

    p_analyze = subs.add_parser("analyze", help="Deep analysis")
    p_analyze.add_argument("slug", type=str)

    p_exec = subs.add_parser("execute", help="Execute arb")
    p_exec.add_argument("slug", type=str)
    p_exec.add_argument("--live", action="store_true")
    p_exec.add_argument("--max-capital", type=float, default=DEFAULT_MAX_CAPITAL)

    p_merge = subs.add_parser("merge", help="Merge on-chain")
    p_merge.add_argument("--execution", type=int, required=True)

    subs.add_parser("status", help="Execution status")

    args = parser.parse_args()

    dispatch = {
        "scan": cmd_scan,
        "analyze": cmd_analyze,
        "execute": cmd_execute,
        "merge": cmd_merge,
        "status": cmd_status,
    }

    if args.command in dispatch:
        dispatch[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
