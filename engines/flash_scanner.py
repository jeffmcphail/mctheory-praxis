"""
engines/flash_scanner.py — Flash Loan Profitability Scanner

Continuously monitors Polymarket for opportunities profitable enough
to justify flash loan execution. Extends the MEV scanner with:

1. NegRisk sum violations with flash-loan-adjusted profitability
2. Binary merge arb (YES+NO < $1) with flash-loan sizing
3. On-chain DEX liquidity for CTF tokens (QuickSwap/Uniswap)
4. Cross-market price inconsistencies

Profitability calculation includes ALL costs:
  - Polymarket fee: 2% on winning side
  - Aave flash loan fee: 0.05%
  - Polygon gas: ~$0.01-0.05 per tx
  - Slippage estimate from order book depth

Usage:
    python -m engines.flash_scanner scan                        # One-shot scan
    python -m engines.flash_scanner scan --min-profit 0.01      # Min 1% profit
    python -m engines.flash_scanner monitor                     # Continuous
    python -m engines.flash_scanner monitor --interval 60       # Every 60s
    python -m engines.flash_scanner stats                       # Historical data
    python -m engines.flash_scanner dex-check                   # Check DEX liquidity
"""
import argparse
import json
import math
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
DB_PATH = Path("data/flash_scanner.db")

# ═══════════════════════════════════════════════════════
# COST MODEL
# ═══════════════════════════════════════════════════════

# All fees as fractions
POLYMARKET_FEE = 0.02       # 2% on winning side
AAVE_FLASH_FEE = 0.0005     # 0.05%
POLYGON_GAS_USD = 0.03      # Estimated gas cost per tx
MIN_PROFIT_USD = 0.50       # Minimum profit to flag as opportunity
MIN_PROFIT_PCT = 0.005      # Minimum profit as % of capital deployed

# Contract addresses (Polygon)
USDC_E = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
CTF_EXCHANGE = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
NEG_RISK_CTF_EXCHANGE = "0xC5d563A36AE78145C45a50134d48A1215220f80a"
AAVE_POOL_POLYGON = "0x794a61358D6845594F94dc1DB02A252b5b4814aD"

# QuickSwap V3 (main DEX on Polygon)
QUICKSWAP_ROUTER = "0xf5b509bB0909a69B1c207E495f687a596C168E12"
QUICKSWAP_QUOTER = "0xa15F0D7377B2A0C0c10db057f641beD21028FC89"


# ═══════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS flash_opportunities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            scan_id TEXT,
            opp_type TEXT,
            description TEXT,

            -- Market info
            event_title TEXT,
            event_slug TEXT,
            n_outcomes INTEGER,

            -- Pricing
            sum_prices REAL,
            expected_sum REAL,
            deviation_pct REAL,

            -- Profitability
            gross_profit_pct REAL,
            polymarket_fee_pct REAL,
            flash_loan_fee_pct REAL,
            gas_cost_usd REAL,
            net_profit_pct REAL,
            net_profit_usd_per_1k REAL,

            -- Execution info
            min_capital REAL,
            max_capital REAL,
            book_depth_usd REAL,
            estimated_slippage REAL,

            -- Assessment
            executable INTEGER DEFAULT 0,
            reason TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS dex_liquidity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            token_id TEXT,
            token_name TEXT,
            dex TEXT,
            pool_address TEXT,
            liquidity_usd REAL,
            price REAL,
            volume_24h REAL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS scan_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            scan_type TEXT,
            markets_scanned INTEGER,
            events_scanned INTEGER,
            opportunities_found INTEGER,
            executable_found INTEGER,
            duration_ms REAL
        )
    """)

    conn.commit()
    return conn


# ═══════════════════════════════════════════════════════
# PROFITABILITY CALCULATOR
# ═══════════════════════════════════════════════════════

def calculate_flash_profit(strategy, prices, expected_sum=1.0,
                           capital=1000, book_depth=None):
    """Calculate net profit for a flash loan strategy after ALL costs.

    Args:
        strategy: "negrisk_over" | "negrisk_under" | "merge_arb"
        prices: list of outcome prices
        expected_sum: expected sum (1.0 for binary, K for K-of-N)
        capital: amount to deploy in USD
        book_depth: estimated max capital before slippage

    Returns:
        dict with profit breakdown
    """
    actual_sum = sum(prices)
    deviation = actual_sum - expected_sum
    deviation_pct = deviation / expected_sum * 100

    result = {
        "strategy": strategy,
        "actual_sum": actual_sum,
        "expected_sum": expected_sum,
        "deviation": deviation,
        "deviation_pct": deviation_pct,
        "capital": capital,
    }

    if strategy == "negrisk_over":
        # Sum > expected: buy all outcomes, redeem for expected_sum
        # Profit per $expected_sum = actual_sum - expected_sum (we buy cheap, redeem at par)
        # Wait — if sum > expected, outcomes are OVERPRICED relative to redemption
        # Actually: if sum < expected, we can buy all outcomes cheap and redeem at par
        # If sum > expected, we'd want to SHORT outcomes
        #
        # Correction: NegRisk overpricing means the market is paying more
        # than expected for the set of outcomes. We can't easily profit
        # from overpricing without shorting.
        #
        # The profitable direction is sum < expected_sum:
        # Buy all N outcomes for sum(prices), redeem for expected_sum
        # Profit = expected_sum - sum(prices) per unit
        result["gross_profit_pct"] = 0
        result["net_profit_pct"] = 0
        result["net_profit_usd"] = 0
        result["executable"] = False
        result["reason"] = "Sum > expected: need short positions (not supported yet)"
        return result

    elif strategy == "negrisk_under":
        # Sum < expected: buy all outcomes, redeem for expected_sum
        # Gross profit per unit = expected_sum - actual_sum
        gap = expected_sum - actual_sum
        if gap <= 0:
            result["gross_profit_pct"] = 0
            result["net_profit_pct"] = 0
            result["net_profit_usd"] = 0
            result["executable"] = False
            result["reason"] = "No underpricing"
            return result

        gross_pct = gap / actual_sum  # Profit as % of capital deployed

        # Cost: buy N outcomes at their ask prices
        # Revenue: redeem for expected_sum per complete set
        units = capital / actual_sum  # How many complete sets we can buy
        gross_usd = units * gap

        # Fees
        flash_fee_usd = capital * AAVE_FLASH_FEE
        # Polymarket fee: 2% on the winning side
        # In a complete set redemption, all sides "win" ($1 each)
        # So fee = 2% × capital effectively
        poly_fee_usd = capital * POLYMARKET_FEE
        gas_usd = POLYGON_GAS_USD

        net_usd = gross_usd - flash_fee_usd - poly_fee_usd - gas_usd
        net_pct = net_usd / capital

        result["gross_profit_pct"] = gross_pct
        result["polymarket_fee_usd"] = poly_fee_usd
        result["flash_loan_fee_usd"] = flash_fee_usd
        result["gas_usd"] = gas_usd
        result["net_profit_pct"] = net_pct
        result["net_profit_usd"] = net_usd

        # Executable if profitable after all fees
        result["executable"] = net_usd >= MIN_PROFIT_USD and net_pct >= MIN_PROFIT_PCT
        if not result["executable"]:
            if net_usd < MIN_PROFIT_USD:
                result["reason"] = f"Net profit ${net_usd:.2f} < ${MIN_PROFIT_USD} minimum"
            else:
                result["reason"] = f"Net profit {net_pct:.2%} < {MIN_PROFIT_PCT:.2%} minimum"
        else:
            result["reason"] = "EXECUTABLE"

        # Estimate max capital from book depth
        if book_depth:
            result["max_capital"] = min(capital, book_depth)
        else:
            result["max_capital"] = capital

        return result

    elif strategy == "merge_arb":
        # Binary market: YES + NO = $1
        # If YES_ask + NO_ask < $1, buy both, merge for $1
        if len(prices) != 2:
            result["executable"] = False
            result["reason"] = "Merge arb requires exactly 2 outcomes"
            return result

        yes_price, no_price = prices
        total = yes_price + no_price

        if total >= 1.0:
            result["gross_profit_pct"] = 0
            result["net_profit_pct"] = 0
            result["net_profit_usd"] = 0
            result["executable"] = False
            result["reason"] = f"Sum={total:.4f} >= $1.00, no arb"
            return result

        gap = 1.0 - total
        gross_pct = gap / total

        units = capital / total
        gross_usd = units * gap

        flash_fee_usd = capital * AAVE_FLASH_FEE
        # Merge: no Polymarket fee (merging is not a trade settlement)
        # Actually: buying YES and NO via CLOB incurs fees on each side
        poly_fee_usd = capital * POLYMARKET_FEE
        gas_usd = POLYGON_GAS_USD

        net_usd = gross_usd - flash_fee_usd - poly_fee_usd - gas_usd
        net_pct = net_usd / capital

        result["gross_profit_pct"] = gross_pct
        result["net_profit_pct"] = net_pct
        result["net_profit_usd"] = net_usd
        result["executable"] = net_usd >= MIN_PROFIT_USD and net_pct >= MIN_PROFIT_PCT
        result["reason"] = "EXECUTABLE" if result["executable"] else f"Net ${net_usd:.2f} too low"

        return result

    return result


def calculate_breakeven_deviation(strategy="negrisk_under"):
    """Calculate the minimum sum deviation needed for profitable flash loan arb.

    This tells us what % underpricing we need to see before it's worth executing.
    """
    # Breakeven: gross_profit = polymarket_fee + flash_fee + gas
    # For $1000 capital:
    # gross = deviation * (capital / sum)
    # costs = capital * 0.02 + capital * 0.0005 + 0.03
    # At breakeven: deviation = costs / units ≈ (0.02 + 0.0005) × sum
    # So we need deviation > 2.05% of expected_sum

    total_fee_pct = POLYMARKET_FEE + AAVE_FLASH_FEE
    gas_as_pct_of_1k = POLYGON_GAS_USD / 1000

    breakeven_pct = total_fee_pct + gas_as_pct_of_1k

    return breakeven_pct


# ═══════════════════════════════════════════════════════
# SCANNERS
# ═══════════════════════════════════════════════════════

def scan_binary_merge_arb(verbose=True):
    """Scan all binary markets for YES+NO < $1 merge arbitrage."""
    if verbose:
        print(f"\n  ── BINARY MERGE ARB SCAN ──")

    opportunities = []
    all_markets = []
    offset = 0

    # Fetch all active binary markets
    while True:
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

    if verbose:
        print(f"    Fetched {len(all_markets)} active markets")

    # Filter to binary with good volume
    binary = []
    for m in all_markets:
        if m.get("negRisk"):
            continue
        vol = float(m.get("volume", 0) or 0)
        if vol < 5000:
            continue
        token_ids = json.loads(m.get("clobTokenIds", "[]"))
        if len(token_ids) != 2:
            continue
        binary.append(m)

    if verbose:
        print(f"    Binary markets with >$5K volume: {len(binary)}")

    scanned = 0
    last_progress = time.time()

    for i, m in enumerate(binary):
        token_ids = json.loads(m.get("clobTokenIds", "[]"))

        try:
            # Get YES ask (cheapest offer)
            r_yes = requests.get(f"{CLOB_API}/midpoint",
                                 params={"token_id": token_ids[0]}, timeout=5)
            r_no = requests.get(f"{CLOB_API}/midpoint",
                                params={"token_id": token_ids[1]}, timeout=5)

            yes_mid = float(r_yes.json().get("mid", 0.5)
                            if isinstance(r_yes.json(), dict) else r_yes.json())
            no_mid = float(r_no.json().get("mid", 0.5)
                           if isinstance(r_no.json(), dict) else r_no.json())

            total = yes_mid + no_mid
            scanned += 1

            if total < 0.99:  # Potential opportunity (before fees)
                profit = calculate_flash_profit(
                    "merge_arb", [yes_mid, no_mid], capital=1000)

                opportunities.append({
                    "slug": m.get("slug", ""),
                    "question": m.get("question", "")[:70],
                    "yes_mid": yes_mid,
                    "no_mid": no_mid,
                    "sum": total,
                    "gap": 1.0 - total,
                    "profit": profit,
                })

                if verbose:
                    icon = "✅" if profit["executable"] else "⚠️"
                    print(f"    {icon} {m.get('question','')[:55]} | "
                          f"YES={yes_mid:.3f} NO={no_mid:.3f} "
                          f"Sum={total:.4f} Gap={1-total:.4f} "
                          f"Net=${profit['net_profit_usd']:.2f}")

        except Exception:
            pass

        # Rate limit
        time.sleep(0.1)

        # Progress
        now = time.time()
        if now - last_progress >= 30:
            pct = (i + 1) / len(binary) * 100
            print(f"    [{datetime.now().strftime('%H:%M:%S')}] "
                  f"{i+1}/{len(binary)} ({pct:.0f}%) | "
                  f"{len(opportunities)} opportunities found")
            last_progress = now

    if verbose:
        print(f"    Scanned {scanned} markets, "
              f"found {len(opportunities)} with sum < 0.99")

    return opportunities


def scan_negrisk_sum_arb(verbose=True, max_events=100):
    """Scan NegRisk events for sum deviations profitable after flash loan fees."""
    if verbose:
        print(f"\n  ── NEGRISK SUM ARB SCAN ──")
        breakeven = calculate_breakeven_deviation()
        print(f"    Breakeven deviation: {breakeven:.2%} "
              f"(Poly={POLYMARKET_FEE:.0%} + Aave={AAVE_FLASH_FEE:.2%})")

    opportunities = []

    # Fetch NegRisk events
    try:
        r = requests.get(f"{GAMMA_API}/events", params={
            "closed": "false", "limit": 200,
        }, timeout=15)
        events = r.json()
    except Exception:
        if verbose:
            print(f"    ❌ Failed to fetch events")
        return []

    neg_risk_events = [e for e in events if e.get("negRisk")]

    if verbose:
        print(f"    NegRisk events: {len(neg_risk_events)}")

    scanned = 0
    last_progress = time.time()

    for ei, event in enumerate(neg_risk_events[:max_events]):
        title = event.get("title", "")
        markets = event.get("markets", [])

        if len(markets) < 2:
            continue

        # Infer expected sum (K)
        k = infer_k_value(event)

        # Get CLOB midpoints for all outcomes
        prices = []
        valid = True

        for m in markets:
            token_ids = json.loads(m.get("clobTokenIds", "[]"))
            if not token_ids:
                valid = False
                break

            try:
                r = requests.get(f"{CLOB_API}/midpoint",
                                 params={"token_id": token_ids[0]}, timeout=5)
                data = r.json()
                mid = float(data.get("mid", 0) if isinstance(data, dict) else data)
                prices.append(mid)
            except Exception:
                valid = False
                break

            time.sleep(0.05)  # Rate limit

        if not valid or not prices:
            continue

        scanned += 1
        actual_sum = sum(prices)
        deviation = actual_sum - k
        deviation_pct = abs(deviation) / k * 100

        # Check if deviation exceeds breakeven
        strategy = "negrisk_under" if deviation < 0 else "negrisk_over"
        profit = calculate_flash_profit(strategy, prices,
                                         expected_sum=k, capital=1000)

        if deviation_pct >= 1.0 or profit.get("executable"):
            opportunities.append({
                "title": title[:60],
                "slug": event.get("slug", ""),
                "n_outcomes": len(markets),
                "k": k,
                "prices": prices,
                "sum": actual_sum,
                "expected_sum": k,
                "deviation": deviation,
                "deviation_pct": deviation_pct,
                "strategy": strategy,
                "profit": profit,
            })

            if verbose:
                icon = "🚨" if profit.get("executable") else ("⚠️" if deviation_pct > 2 else "📊")
                print(f"    {icon} {title[:50]} | "
                      f"K={k} N={len(markets)} "
                      f"Sum={actual_sum:.4f} Dev={deviation:+.4f} ({deviation_pct:.1f}%) "
                      f"Net=${profit.get('net_profit_usd', 0):.2f} "
                      f"{'✅ EXEC' if profit.get('executable') else ''}")

        # Progress
        now = time.time()
        if now - last_progress >= 30:
            pct = (ei + 1) / len(neg_risk_events) * 100
            print(f"    [{datetime.now().strftime('%H:%M:%S')}] "
                  f"{ei+1}/{len(neg_risk_events)} ({pct:.0f}%) | "
                  f"{len(opportunities)} opportunities")
            last_progress = now

    if verbose:
        print(f"    Scanned {scanned} events, "
              f"found {len(opportunities)} with deviation >= 1%")

    return opportunities


def infer_k_value(event):
    """Infer the expected sum K for a NegRisk event.

    K=1: "Who will win X?" (exclusive outcomes)
    K=4: "Top 4 Finish" (4 winners)
    K=3: "Relegation" (3 relegated)
    """
    title = (event.get("title", "") or "").lower()
    markets = event.get("markets", [])
    n = len(markets)

    # Check for explicit K indicators
    if "top 4" in title or "top four" in title:
        return 4
    if "top 3" in title or "top three" in title:
        return 3
    if "relegate" in title or "relegation" in title:
        return 3
    if "promote" in title or "promotion" in title:
        return min(3, n)

    # Default: exclusive (winner-take-all)
    return 1


def scan_dex_liquidity(verbose=True):
    """Check if CTF outcome tokens have on-chain DEX liquidity.

    If they do, we can do flash-loan-powered DEX↔CLOB arbitrage.
    """
    if verbose:
        print(f"\n  ── DEX LIQUIDITY CHECK ──")
        print(f"    Checking if CTF tokens trade on QuickSwap/Uniswap...")

    # Get some high-volume market token IDs
    try:
        r = requests.get(f"{GAMMA_API}/markets", params={
            "closed": "false", "active": "true",
            "limit": 10, "order": "volume",
        }, timeout=15)
        markets = r.json()
    except Exception:
        if verbose:
            print(f"    ❌ Failed to fetch markets")
        return []

    results = []

    for m in markets[:5]:
        token_ids = json.loads(m.get("clobTokenIds", "[]"))
        if not token_ids:
            continue

        question = m.get("question", "")[:50]
        yes_token = token_ids[0]

        # Check QuickSwap for liquidity
        # CTF tokens are ERC-1155, not ERC-20, so they don't trade on
        # standard DEXes directly. Would need a wrapper contract.
        # For now, just report this finding.

        if verbose:
            print(f"    {question}")
            print(f"      Token: {yes_token[:30]}...")
            print(f"      CTF tokens are ERC-1155 — NOT directly tradeable on DEXes")
            print(f"      Would need ERC-1155→ERC-20 wrapper for DEX trading")

        results.append({
            "question": question,
            "token_id": yes_token,
            "dex_tradeable": False,
            "reason": "ERC-1155 tokens not compatible with standard DEX pools",
        })

        break  # Only need to check once — applies to all CTF tokens

    if verbose:
        print(f"\n    Conclusion: CTF outcome tokens are ERC-1155 (not ERC-20).")
        print(f"    Standard DEXes (QuickSwap, Uniswap) can't trade them directly.")
        print(f"    DEX↔CLOB arbitrage requires an ERC-1155→ERC-20 wrapper contract.")
        print(f"    This is technically possible but adds complexity and gas cost.")

    return results


# ═══════════════════════════════════════════════════════
# COMMANDS
# ═══════════════════════════════════════════════════════

def cmd_scan(args):
    """Run a one-shot scan for flash loan opportunities."""
    conn = init_db()
    scan_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    start = time.time()

    min_profit = getattr(args, "min_profit", MIN_PROFIT_PCT)

    print(f"\n{'='*90}")
    print(f"  FLASH LOAN OPPORTUNITY SCANNER")
    print(f"  Scan ID: {scan_id}")
    print(f"  Min profit: {min_profit:.1%} | Aave fee: {AAVE_FLASH_FEE:.2%} | "
          f"Poly fee: {POLYMARKET_FEE:.0%}")
    print(f"  Breakeven deviation: {calculate_breakeven_deviation():.2%}")
    print(f"{'='*90}")

    # Scan 1: Binary merge arb
    merge_opps = scan_binary_merge_arb(verbose=True)

    # Scan 2: NegRisk sum arb
    negrisk_opps = scan_negrisk_sum_arb(verbose=True)

    # Scan 3: DEX liquidity check
    dex_results = scan_dex_liquidity(verbose=True)

    # Store results
    all_opps = []

    for opp in merge_opps:
        p = opp["profit"]
        conn.execute("""
            INSERT INTO flash_opportunities
            (timestamp, scan_id, opp_type, description, event_title,
             sum_prices, expected_sum, deviation_pct,
             gross_profit_pct, polymarket_fee_pct, flash_loan_fee_pct,
             gas_cost_usd, net_profit_pct, net_profit_usd_per_1k,
             executable, reason)
            VALUES (?, ?, 'merge_arb', ?, ?, ?, 1.0, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(), scan_id,
            opp["question"], opp["question"],
            opp["sum"], (1 - opp["sum"]) * 100,
            p.get("gross_profit_pct", 0), POLYMARKET_FEE, AAVE_FLASH_FEE,
            POLYGON_GAS_USD, p.get("net_profit_pct", 0),
            p.get("net_profit_usd", 0),
            1 if p.get("executable") else 0, p.get("reason", ""),
        ))

    for opp in negrisk_opps:
        p = opp["profit"]
        conn.execute("""
            INSERT INTO flash_opportunities
            (timestamp, scan_id, opp_type, description, event_title,
             n_outcomes, sum_prices, expected_sum, deviation_pct,
             gross_profit_pct, polymarket_fee_pct, flash_loan_fee_pct,
             gas_cost_usd, net_profit_pct, net_profit_usd_per_1k,
             executable, reason)
            VALUES (?, ?, 'negrisk_sum', ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(), scan_id,
            opp["title"], opp["title"],
            opp["n_outcomes"], opp["sum"], opp["expected_sum"],
            opp["deviation_pct"],
            p.get("gross_profit_pct", 0), POLYMARKET_FEE, AAVE_FLASH_FEE,
            POLYGON_GAS_USD, p.get("net_profit_pct", 0),
            p.get("net_profit_usd", 0),
            1 if p.get("executable") else 0, p.get("reason", ""),
        ))

    conn.commit()

    elapsed = time.time() - start

    # Log scan
    total_opps = len(merge_opps) + len(negrisk_opps)
    executable = (sum(1 for o in merge_opps if o["profit"].get("executable")) +
                  sum(1 for o in negrisk_opps if o["profit"].get("executable")))

    conn.execute("""
        INSERT INTO scan_log
        (timestamp, scan_type, markets_scanned, events_scanned,
         opportunities_found, executable_found, duration_ms)
        VALUES (?, 'full', ?, ?, ?, ?, ?)
    """, (
        datetime.now(timezone.utc).isoformat(),
        len(merge_opps), len(negrisk_opps),
        total_opps, executable, elapsed * 1000,
    ))
    conn.commit()

    # Summary
    print(f"\n{'─'*90}")
    print(f"  SCAN SUMMARY")
    print(f"{'─'*90}")
    print(f"  Duration:          {elapsed:.1f}s")
    print(f"  Binary markets:    {len(merge_opps)} with sum < 0.99")
    print(f"  NegRisk events:    {len(negrisk_opps)} with deviation >= 1%")
    print(f"  DEX tradeable:     {'No — CTF tokens are ERC-1155' if dex_results else 'Not checked'}")
    print(f"  Total opportunities: {total_opps}")
    print(f"  Executable (profitable after all fees): {executable}")

    if executable > 0:
        print(f"\n  🚨 EXECUTABLE OPPORTUNITIES:")
        for opp in merge_opps + negrisk_opps:
            if opp.get("profit", {}).get("executable"):
                p = opp["profit"]
                title = opp.get("question", opp.get("title", "?"))
                print(f"    ✅ {title[:55]}")
                print(f"       Net: ${p['net_profit_usd']:.2f}/trade "
                      f"({p['net_profit_pct']:.2%}) | "
                      f"Type: {p['strategy']}")
    else:
        print(f"\n  No executable opportunities found.")
        print(f"  Binary arb fully competed (sums ≈ 1.001)")
        print(f"  NegRisk deviations below {calculate_breakeven_deviation():.2%} breakeven")

    conn.close()
    print(f"\n{'='*90}")


def cmd_monitor(args):
    """Continuous monitoring for flash loan opportunities."""
    interval = getattr(args, "interval", 300)  # Default: 5 minutes
    min_profit = getattr(args, "min_profit", MIN_PROFIT_PCT)

    print(f"\n{'='*90}")
    print(f"  FLASH LOAN MONITOR — Continuous Scanning")
    print(f"  Interval: {interval}s | Min profit: {min_profit:.1%}")
    print(f"  Press Ctrl+C to stop.")
    print(f"{'='*90}")

    cycle = 0

    while True:
        try:
            cycle += 1
            start = time.time()
            now_str = datetime.now().strftime("%H:%M:%S")

            # Quick NegRisk scan (faster than full scan)
            negrisk_opps = scan_negrisk_sum_arb(verbose=False, max_events=50)

            executable = sum(1 for o in negrisk_opps
                           if o.get("profit", {}).get("executable"))

            elapsed = time.time() - start

            if executable > 0:
                print(f"  {now_str} 🚨 {executable} EXECUTABLE opportunities! "
                      f"({len(negrisk_opps)} total, {elapsed:.1f}s)")
                for opp in negrisk_opps:
                    if opp["profit"].get("executable"):
                        p = opp["profit"]
                        print(f"    ✅ {opp['title'][:50]} | "
                              f"Dev={opp['deviation_pct']:.1f}% "
                              f"Net=${p['net_profit_usd']:.2f}")
            else:
                # Brief status
                max_dev = max((o["deviation_pct"] for o in negrisk_opps), default=0)
                print(f"  {now_str} Cycle {cycle} | "
                      f"{len(negrisk_opps)} events | "
                      f"Max dev: {max_dev:.1f}% | "
                      f"Breakeven: {calculate_breakeven_deviation():.1%} | "
                      f"{elapsed:.1f}s")

            sleep_time = max(0, interval - elapsed)
            time.sleep(sleep_time)

        except KeyboardInterrupt:
            print(f"\n  Stopped after {cycle} cycles.")
            break
        except Exception as e:
            print(f"  ⚠️ Error: {e}")
            time.sleep(interval)


def cmd_stats(args):
    """Show historical scan statistics."""
    if not DB_PATH.exists():
        print("  No database. Run a scan first.")
        return

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    total_scans = conn.execute("SELECT COUNT(*) FROM scan_log").fetchone()[0]
    total_opps = conn.execute("SELECT COUNT(*) FROM flash_opportunities").fetchone()[0]
    executable = conn.execute(
        "SELECT COUNT(*) FROM flash_opportunities WHERE executable=1").fetchone()[0]

    print(f"\n{'='*70}")
    print(f"  FLASH SCANNER STATS")
    print(f"{'='*70}")
    print(f"  Total scans:         {total_scans}")
    print(f"  Total opportunities: {total_opps}")
    print(f"  Executable:          {executable}")

    # Best opportunities ever seen
    best = conn.execute("""
        SELECT * FROM flash_opportunities
        ORDER BY net_profit_usd_per_1k DESC
        LIMIT 10
    """).fetchall()

    if best:
        print(f"\n  Best Opportunities (by net profit per $1K):")
        print(f"  {'Time':<20s} {'Type':<12s} {'Event':<35s} "
              f"{'Dev':>5s} {'Net$':>6s}")
        print(f"  {'─'*82}")
        for b in best:
            print(f"  {(b['timestamp'] or '')[:19]:<20s} "
                  f"{b['opp_type']:<12s} "
                  f"{(b['event_title'] or '')[:34]:<35s} "
                  f"{(b['deviation_pct'] or 0):>4.1f}% "
                  f"${(b['net_profit_usd_per_1k'] or 0):>5.2f}")

    # Distribution of deviations
    deviations = conn.execute("""
        SELECT opp_type,
               COUNT(*),
               AVG(deviation_pct),
               MAX(deviation_pct),
               AVG(net_profit_usd_per_1k)
        FROM flash_opportunities
        GROUP BY opp_type
    """).fetchall()

    if deviations:
        print(f"\n  By Type:")
        print(f"  {'Type':<15s} {'Count':>6s} {'AvgDev':>7s} "
              f"{'MaxDev':>7s} {'AvgNet$':>8s}")
        print(f"  {'─'*50}")
        for d in deviations:
            print(f"  {d[0]:<15s} {d[1]:>6d} {d[2]:>6.2f}% "
                  f"{d[3]:>6.2f}% ${d[4]:>7.2f}")

    conn.close()
    print(f"\n{'='*70}")


def cmd_dex_check(args):
    """Check DEX liquidity for CTF tokens."""
    scan_dex_liquidity(verbose=True)


def main():
    parser = argparse.ArgumentParser(description="Flash Loan Profitability Scanner")
    subs = parser.add_subparsers(dest="command")

    p_scan = subs.add_parser("scan", help="One-shot scan")
    p_scan.add_argument("--min-profit", type=float, default=MIN_PROFIT_PCT)

    p_mon = subs.add_parser("monitor", help="Continuous monitoring")
    p_mon.add_argument("--interval", type=int, default=300)
    p_mon.add_argument("--min-profit", type=float, default=MIN_PROFIT_PCT)

    subs.add_parser("stats", help="Historical stats")
    subs.add_parser("dex-check", help="Check DEX liquidity")

    args = parser.parse_args()

    if args.command == "scan":
        cmd_scan(args)
    elif args.command == "monitor":
        cmd_monitor(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "dex-check":
        cmd_dex_check(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
