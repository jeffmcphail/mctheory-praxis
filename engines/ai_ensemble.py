"""
engines/ai_ensemble.py — AI Ensemble Probability Engine

Queries multiple LLMs for their probability estimate on Polymarket questions,
builds a consensus probability, and compares it to market prices. When the
AI consensus diverges >15% from market price, it flags a potential edge.

Architecture:
    1. FETCH: Pull active high-volume markets from Polymarket
    2. QUERY: Ask each LLM "What is the probability of X?"
    3. AGGREGATE: Weighted consensus across providers
    4. COMPARE: AI consensus vs CLOB market price
    5. FLAG: Divergences > threshold → trading signal

Providers:
    - DeepSeek (V3.2) — cheap, fast, good reasoning
    - Anthropic (Claude) — expensive, highest quality
    - Groq (Llama) — free tier, fast, decent

Bridges AI Agent Factory's ProviderBridge concept into Praxis.
Eventually uses QPT evolution to optimize prompt loadings.

Usage:
    python -m engines.ai_ensemble scan                         # Scan top markets
    python -m engines.ai_ensemble scan --top 50 --threshold 10 # Custom params
    python -m engines.ai_ensemble ask "Will BTC hit 100k?"     # Single question
    python -m engines.ai_ensemble divergences                  # Show all divergences
    python -m engines.ai_ensemble monitor                      # Continuous scanning
    python -m engines.ai_ensemble calibration                  # How accurate are the LLMs?
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
DB_PATH = Path("data/ai_ensemble.db")

# Provider config
PROVIDERS = {
    "deepseek": {
        "env_key": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "weight": 1.0,        # Consensus weight
        "cost_per_1k": 0.001,  # Rough cost per 1K tokens
        "tier": "cheap",
    },
    "groq": {
        "env_key": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.1-70b-versatile",
        "weight": 0.8,
        "cost_per_1k": 0.0,
        "tier": "free",
    },
    "anthropic": {
        "env_key": "ANTHROPIC_API_KEY",
        "base_url": None,  # Uses native SDK
        "model": "claude-sonnet-4-20250514",
        "weight": 1.5,        # Higher weight — better reasoning
        "cost_per_1k": 0.003,
        "tier": "premium",
    },
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "base_url": None,
        "model": "gpt-4o-mini",
        "weight": 1.0,
        "cost_per_1k": 0.0015,
        "tier": "mid",
    },
}

# Ensemble config
DEFAULT_THRESHOLD = 15.0    # Flag divergences > 15%
DEFAULT_TOP_N = 20          # Scan top N markets
MIN_VOLUME = 50000          # Minimum market volume
RATE_LIMIT_SECS = 1.0       # Between API calls

SYSTEM_PROMPT = """You are a probability estimation expert. Given a prediction market question, estimate the probability that the outcome will be YES.

CRITICAL RULES:
1. Respond with ONLY a JSON object, no markdown, no explanation
2. Your probability MUST be between 0.01 and 0.99
3. Consider the current date: {date}
4. Factor in all publicly available information
5. Be well-calibrated — a 70% estimate should resolve YES 70% of the time

Response format:
{{"probability": 0.XX, "confidence": 0.XX, "reasoning": "brief one-line explanation"}}

Where:
- probability: your best estimate (0.01 to 0.99)
- confidence: how confident you are in your estimate (0.0 to 1.0)
- reasoning: brief explanation (one line, under 100 chars)"""


# ═══════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS estimates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            scan_id TEXT,
            market_slug TEXT,
            question TEXT,
            provider TEXT,
            model TEXT,
            probability REAL,
            confidence REAL,
            reasoning TEXT,
            latency_ms REAL,
            error TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS consensus (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            scan_id TEXT,
            market_slug TEXT,
            question TEXT,
            market_price REAL,
            consensus_prob REAL,
            n_providers INTEGER,
            providers_used TEXT,
            divergence_pct REAL,
            divergence_direction TEXT,
            flagged INTEGER DEFAULT 0
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS resolution_tracking (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_slug TEXT UNIQUE,
            question TEXT,
            consensus_at_flag REAL,
            market_price_at_flag REAL,
            flagged_at TEXT,
            resolved INTEGER DEFAULT 0,
            resolution TEXT,
            resolved_at TEXT,
            consensus_correct INTEGER
        )
    """)

    conn.commit()
    return conn


# ═══════════════════════════════════════════════════════
# LLM PROVIDERS
# ═══════════════════════════════════════════════════════

def get_available_providers():
    """Check which LLM providers have API keys configured."""
    available = {}
    for name, config in PROVIDERS.items():
        key = os.getenv(config["env_key"], "")
        if key:
            available[name] = config
    return available


def query_provider_openai_compatible(name, config, question):
    """Query an OpenAI-compatible provider (DeepSeek, Groq, OpenAI)."""
    try:
        from openai import OpenAI
    except ImportError:
        return None, "openai package not installed"

    key = os.getenv(config["env_key"], "")
    if not key:
        return None, f"No {config['env_key']}"

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    system = SYSTEM_PROMPT.format(date=today)

    try:
        kwargs = {"api_key": key}
        if config["base_url"]:
            kwargs["base_url"] = config["base_url"]

        client = OpenAI(**kwargs)
        start = time.time()

        resp = client.chat.completions.create(
            model=config["model"],
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Question: {question}"},
            ],
            temperature=0.3,
            max_tokens=200,
        )

        latency = (time.time() - start) * 1000
        text = resp.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()

        result = json.loads(text)
        prob = float(result.get("probability", 0.5))
        prob = max(0.01, min(0.99, prob))
        conf = float(result.get("confidence", 0.5))
        reasoning = result.get("reasoning", "")[:100]

        return {
            "probability": prob,
            "confidence": conf,
            "reasoning": reasoning,
            "latency_ms": latency,
        }, None

    except json.JSONDecodeError:
        # Try to extract probability from raw text
        try:
            import re
            match = re.search(r'0\.\d+', text)
            if match:
                prob = float(match.group())
                return {
                    "probability": max(0.01, min(0.99, prob)),
                    "confidence": 0.3,
                    "reasoning": "parsed from raw text",
                    "latency_ms": latency,
                }, None
        except Exception:
            pass
        return None, f"JSON parse failed: {text[:50]}"

    except Exception as e:
        return None, str(e)[:80]


def query_provider_anthropic(config, question):
    """Query Anthropic's Claude API."""
    key = os.getenv(config["env_key"], "")
    if not key:
        return None, "No ANTHROPIC_API_KEY"

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    system = SYSTEM_PROMPT.format(date=today)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        start = time.time()

        resp = client.messages.create(
            model=config["model"],
            max_tokens=200,
            system=system,
            messages=[
                {"role": "user", "content": f"Question: {question}"},
            ],
            temperature=0.3,
        )

        latency = (time.time() - start) * 1000
        text = resp.content[0].text.strip()
        text = text.replace("```json", "").replace("```", "").strip()

        result = json.loads(text)
        prob = float(result.get("probability", 0.5))
        prob = max(0.01, min(0.99, prob))

        return {
            "probability": prob,
            "confidence": float(result.get("confidence", 0.5)),
            "reasoning": result.get("reasoning", "")[:100],
            "latency_ms": latency,
        }, None

    except Exception as e:
        return None, str(e)[:80]


def query_provider(name, config, question):
    """Route to the correct provider query method."""
    if name == "anthropic":
        return query_provider_anthropic(config, question)
    else:
        return query_provider_openai_compatible(name, config, question)


def query_ensemble(question, providers=None, verbose=False):
    """Query all available providers and build consensus.

    Returns:
        dict with consensus probability, individual estimates, etc.
    """
    available = providers or get_available_providers()

    if not available:
        return {"error": "No LLM providers configured"}

    estimates = []
    errors = []

    for name, config in available.items():
        if verbose:
            print(f"      Querying {name} ({config['model']})...", end=" ", flush=True)

        result, error = query_provider(name, config, question)

        if result:
            result["provider"] = name
            result["model"] = config["model"]
            result["weight"] = config["weight"]
            estimates.append(result)
            if verbose:
                print(f"{result['probability']:.0%} "
                      f"(conf={result['confidence']:.0%}, "
                      f"{result['latency_ms']:.0f}ms)")
        else:
            errors.append({"provider": name, "error": error})
            if verbose:
                print(f"❌ {error}")

        time.sleep(RATE_LIMIT_SECS)

    if not estimates:
        return {"error": "All providers failed", "errors": errors}

    # Weighted consensus
    total_weight = sum(e["weight"] * e["confidence"] for e in estimates)
    if total_weight > 0:
        consensus = sum(
            e["probability"] * e["weight"] * e["confidence"]
            for e in estimates
        ) / total_weight
    else:
        consensus = sum(e["probability"] for e in estimates) / len(estimates)

    # Spread (disagreement between providers)
    probs = [e["probability"] for e in estimates]
    spread = max(probs) - min(probs) if len(probs) > 1 else 0

    return {
        "consensus": consensus,
        "n_providers": len(estimates),
        "spread": spread,
        "estimates": estimates,
        "errors": errors,
    }


# ═══════════════════════════════════════════════════════
# COMMANDS
# ═══════════════════════════════════════════════════════

def cmd_ask(args):
    """Ask all LLMs a single question."""
    question = args.question
    available = get_available_providers()

    print(f"\n{'='*70}")
    print(f"  AI ENSEMBLE — Single Question")
    print(f"  Providers: {', '.join(available.keys())}")
    print(f"{'='*70}")
    print(f"\n  Q: {question}\n")

    result = query_ensemble(question, available, verbose=True)

    if "error" in result:
        print(f"\n  ❌ {result['error']}")
        return

    print(f"\n  ── CONSENSUS ──")
    print(f"  Probability:  {result['consensus']:.1%}")
    print(f"  Providers:    {result['n_providers']}")
    print(f"  Spread:       {result['spread']:.1%} "
          f"({'tight' if result['spread'] < 0.1 else 'wide'})")

    print(f"\n{'='*70}")


def cmd_scan(args):
    """Scan top markets and compare AI consensus vs market prices."""
    top_n = getattr(args, "top", DEFAULT_TOP_N)
    threshold = getattr(args, "threshold", DEFAULT_THRESHOLD)
    cheap_only = getattr(args, "cheap", False)

    conn = init_db()
    available = get_available_providers()

    if cheap_only:
        available = {k: v for k, v in available.items()
                     if v["tier"] in ("cheap", "free")}

    if not available:
        print("  ❌ No LLM providers configured. Set API keys in .env")
        return

    scan_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*90}")
    print(f"  AI ENSEMBLE PROBABILITY SCAN")
    print(f"  Scan ID: {scan_id}")
    print(f"  Providers: {', '.join(available.keys())}")
    print(f"  Markets: top {top_n} by volume | Threshold: {threshold}%")
    if cheap_only:
        print(f"  Mode: CHEAP ONLY (no Anthropic)")
    print(f"{'='*90}")

    # Fetch top markets
    all_markets = []
    offset = 0
    while len(all_markets) < top_n * 2:
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

    # Filter: binary, good volume, not near resolution
    candidates = []
    for m in all_markets:
        vol = float(m.get("volume", 0) or 0)
        if vol < MIN_VOLUME:
            continue
        # Skip NegRisk sub-markets (use event-level questions instead)
        if m.get("negRisk"):
            continue
        token_ids = json.loads(m.get("clobTokenIds", "[]"))
        if len(token_ids) != 2:
            continue
        # Get current price
        try:
            r = requests.get(f"{CLOB_API}/midpoint",
                             params={"token_id": token_ids[0]}, timeout=5)
            data = r.json()
            mid = float(data.get("mid", 0.5) if isinstance(data, dict) else data)
        except Exception:
            mid = 0.5

        # Skip markets near 0 or 1 (already resolved effectively)
        if mid < 0.05 or mid > 0.95:
            continue

        candidates.append({
            "slug": m.get("slug", ""),
            "question": m.get("question", ""),
            "volume": vol,
            "market_price": mid,
            "token_id": token_ids[0],
        })

    candidates.sort(key=lambda x: -x["volume"])
    candidates = candidates[:top_n]

    print(f"\n  Scanning {len(candidates)} markets...\n")

    divergences = []
    last_progress = time.time()

    for i, market in enumerate(candidates):
        question = market["question"]
        market_price = market["market_price"]
        slug = market["slug"]

        print(f"  [{i+1}/{len(candidates)}] {question[:60]}")
        print(f"    Market price: {market_price:.1%}")

        # Query ensemble
        result = query_ensemble(question, available, verbose=True)

        if "error" in result:
            print(f"    ❌ {result['error']}\n")
            continue

        consensus = result["consensus"]
        divergence = (consensus - market_price) * 100  # In percentage points

        # Store individual estimates
        now = datetime.now(timezone.utc).isoformat()
        for est in result["estimates"]:
            conn.execute("""
                INSERT INTO estimates
                (timestamp, scan_id, market_slug, question, provider, model,
                 probability, confidence, reasoning, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now, scan_id, slug, question[:200],
                est["provider"], est["model"],
                est["probability"], est["confidence"],
                est["reasoning"], est["latency_ms"],
            ))

        # Store consensus
        direction = "AI_HIGHER" if divergence > 0 else "AI_LOWER"
        flagged = abs(divergence) >= threshold

        conn.execute("""
            INSERT INTO consensus
            (timestamp, scan_id, market_slug, question, market_price,
             consensus_prob, n_providers, providers_used,
             divergence_pct, divergence_direction, flagged)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now, scan_id, slug, question[:200], market_price,
            consensus, result["n_providers"],
            ",".join(e["provider"] for e in result["estimates"]),
            divergence, direction, 1 if flagged else 0,
        ))

        icon = "🚨" if flagged else "  "
        print(f"    Consensus: {consensus:.1%} | "
              f"Divergence: {divergence:+.1f}pp | "
              f"Spread: {result['spread']:.1%} {icon}")
        print()

        if flagged:
            divergences.append({
                "slug": slug,
                "question": question[:60],
                "market_price": market_price,
                "consensus": consensus,
                "divergence": divergence,
                "direction": direction,
                "spread": result["spread"],
                "n_providers": result["n_providers"],
            })

            # Track for resolution
            conn.execute("""
                INSERT OR IGNORE INTO resolution_tracking
                (market_slug, question, consensus_at_flag, market_price_at_flag,
                 flagged_at)
                VALUES (?, ?, ?, ?, ?)
            """, (slug, question[:200], consensus, market_price, now))

        conn.commit()

    # Summary
    print(f"\n{'─'*90}")
    print(f"  SCAN RESULTS")
    print(f"{'─'*90}")
    print(f"  Markets scanned:     {len(candidates)}")
    print(f"  Divergences flagged: {len(divergences)} (>{threshold}%)")

    if divergences:
        print(f"\n  🚨 FLAGGED DIVERGENCES:")
        print(f"  {'Market':<50s} {'Mkt':>5s} {'AI':>5s} {'Div':>7s} {'Dir'}")
        print(f"  {'─'*80}")

        for d in sorted(divergences, key=lambda x: -abs(x["divergence"])):
            print(f"  {d['question']:<50s} "
                  f"{d['market_price']:>4.0%} {d['consensus']:>4.0%} "
                  f"{d['divergence']:>+6.1f}pp "
                  f"{'📈 AI higher' if d['direction'] == 'AI_HIGHER' else '📉 AI lower'}")

        print(f"\n  Interpretation:")
        print(f"  AI_HIGHER = AI thinks YES is more likely than market → buy YES")
        print(f"  AI_LOWER  = AI thinks YES is less likely than market → buy NO")
    else:
        print(f"\n  No significant divergences found.")
        print(f"  Markets appear fairly priced relative to AI consensus.")

    conn.close()
    print(f"\n{'='*90}")


def cmd_divergences(args):
    """Show all historical divergences."""
    if not DB_PATH.exists():
        print("  No data. Run scan first.")
        return

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    flagged = conn.execute("""
        SELECT * FROM consensus WHERE flagged=1
        ORDER BY ABS(divergence_pct) DESC
    """).fetchall()

    print(f"\n{'='*70}")
    print(f"  ALL FLAGGED DIVERGENCES ({len(flagged)})")
    print(f"{'='*70}")

    if not flagged:
        print(f"\n  No divergences flagged yet.")
    else:
        print(f"\n  {'Time':<20s} {'Market':<40s} {'Mkt':>5s} "
              f"{'AI':>5s} {'Div':>7s}")
        print(f"  {'─'*80}")
        for f in flagged:
            print(f"  {f['timestamp'][:19]:<20s} "
                  f"{f['question'][:39]:<40s} "
                  f"{f['market_price']:>4.0%} {f['consensus_prob']:>4.0%} "
                  f"{f['divergence_pct']:>+6.1f}pp")

    conn.close()
    print(f"\n{'='*70}")


def cmd_calibration(args):
    """Check AI calibration against resolved markets."""
    if not DB_PATH.exists():
        print("  No data.")
        return

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # Check resolved markets
    tracked = conn.execute("""
        SELECT * FROM resolution_tracking WHERE resolved=1
    """).fetchall()

    print(f"\n{'='*70}")
    print(f"  AI CALIBRATION REPORT")
    print(f"{'='*70}")

    if not tracked:
        print(f"\n  No resolved markets yet. Calibration data will accumulate")
        print(f"  as flagged markets resolve.")

        # Show pending
        pending = conn.execute(
            "SELECT COUNT(*) FROM resolution_tracking WHERE resolved=0"
        ).fetchone()[0]
        print(f"  Pending resolution: {pending} markets")
    else:
        correct = sum(1 for t in tracked if t["consensus_correct"])
        total = len(tracked)
        accuracy = correct / total if total > 0 else 0

        print(f"\n  Resolved markets: {total}")
        print(f"  AI correct:       {correct} ({accuracy:.0%})")

        # Brier score by provider
        estimates = conn.execute("""
            SELECT provider, AVG(probability) as avg_prob,
                   COUNT(*) as n
            FROM estimates
            GROUP BY provider
        """).fetchall()

        if estimates:
            print(f"\n  Provider Stats:")
            print(f"  {'Provider':<15s} {'AvgProb':>8s} {'N':>6s}")
            print(f"  {'─'*30}")
            for e in estimates:
                print(f"  {e['provider']:<15s} {e['avg_prob']:>7.1%} {e['n']:>6d}")

    conn.close()
    print(f"\n{'='*70}")


def cmd_monitor(args):
    """Continuous scanning for divergences."""
    interval = getattr(args, "interval", 1800)  # Default: 30 min
    threshold = getattr(args, "threshold", DEFAULT_THRESHOLD)
    top_n = getattr(args, "top", 10)

    print(f"\n{'='*70}")
    print(f"  AI ENSEMBLE MONITOR")
    print(f"  Interval: {interval}s | Top {top_n} markets | "
          f"Threshold: {threshold}%")
    print(f"  Press Ctrl+C to stop.")
    print(f"{'='*70}")

    # Build a fake args object for cmd_scan
    class ScanArgs:
        pass

    scan_args = ScanArgs()
    scan_args.top = top_n
    scan_args.threshold = threshold
    scan_args.cheap = True  # Use cheap providers for monitoring

    cycle = 0
    while True:
        try:
            cycle += 1
            print(f"\n  ── Cycle {cycle} [{datetime.now().strftime('%H:%M:%S')}] ──")
            cmd_scan(scan_args)
            time.sleep(interval)
        except KeyboardInterrupt:
            print(f"\n  Stopped after {cycle} cycles.")
            break
        except Exception as e:
            print(f"  ⚠️ Error: {e}")
            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="AI Ensemble Probability Engine")
    subs = parser.add_subparsers(dest="command")

    p_ask = subs.add_parser("ask", help="Ask a single question")
    p_ask.add_argument("question", type=str)

    p_scan = subs.add_parser("scan", help="Scan top markets")
    p_scan.add_argument("--top", type=int, default=DEFAULT_TOP_N)
    p_scan.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p_scan.add_argument("--cheap", action="store_true",
                        help="Use only cheap/free providers")

    subs.add_parser("divergences", help="Show flagged divergences")
    subs.add_parser("calibration", help="AI calibration report")

    p_mon = subs.add_parser("monitor", help="Continuous scanning")
    p_mon.add_argument("--interval", type=int, default=1800)
    p_mon.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p_mon.add_argument("--top", type=int, default=10)

    args = parser.parse_args()

    if args.command == "ask":
        cmd_ask(args)
    elif args.command == "scan":
        cmd_scan(args)
    elif args.command == "divergences":
        cmd_divergences(args)
    elif args.command == "calibration":
        cmd_calibration(args)
    elif args.command == "monitor":
        cmd_monitor(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
