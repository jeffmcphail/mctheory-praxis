"""
engines/correlation_engine.py — NLP Market Correlation Engine

Parses Polymarket questions to extract entities, events, and thresholds.
Clusters related markets and identifies potential Bayesian mispricings.

Architecture:
    Layer 1: PARSE — extract structured data from free-text questions
    Layer 2: CLUSTER — group markets by shared entities/causal links  
    Layer 3: ANALYZE — compute conditional probabilities (future)
    Layer 4: SIGNAL — generate tradeable signals (future)

Usage:
    python -m engines.correlation_engine scan
    python -m engines.correlation_engine clusters --grep "inflation"
    python -m engines.correlation_engine chain --topic "oil"
"""
import argparse
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timezone

import requests

GAMMA_API = "https://gamma-api.polymarket.com"

# ──────────────────────────────────────────────────────────────
# NLP TOKENIZER — word-boundary matching with exclusions
# ──────────────────────────────────────────────────────────────

def make_pattern(keyword):
    """Build a regex that matches keyword at word boundaries."""
    escaped = re.escape(keyword.strip())
    return re.compile(r'(?<![a-zA-Z])' + escaped + r'(?![a-zA-Z])', re.IGNORECASE)


class EntityMatcher:
    """NLP entity matcher with word-boundary tokenization and exclusion lists."""

    def __init__(self, entity_dict, exclusions=None):
        """
        entity_dict: {"entity_name": ["keyword1", "keyword2", ...]}
        exclusions: {"keyword": ["context_word_that_invalidates"]}
        """
        self.entities = {}
        self.exclusions = exclusions or {}

        for entity, keywords in entity_dict.items():
            self.entities[entity] = [
                (kw, make_pattern(kw)) for kw in keywords
            ]

    def match(self, text):
        """Return list of matched entity names."""
        matched = []
        text_lower = text.lower()

        for entity, kw_patterns in self.entities.items():
            for kw, pattern in kw_patterns:
                if pattern.search(text_lower):
                    # Check exclusions
                    excluded = False
                    if kw in self.exclusions:
                        for exc_word in self.exclusions[kw]:
                            if exc_word.lower() in text_lower:
                                excluded = True
                                break
                    if not excluded:
                        matched.append(entity)
                        break  # One match per entity is enough

        return matched


# ──────────────────────────────────────────────────────────────
# ENTITY DEFINITIONS
# ──────────────────────────────────────────────────────────────

COUNTRIES = {
    "us": ["united states", "u.s.", "american", "usa"],
    "iran": ["iran", "iranian", "tehran", "hormuz", "persian gulf"],
    "israel": ["israel", "israeli", "gaza"],
    "china": ["china", "chinese", "beijing", "xi jinping"],
    "russia": ["russia", "russian", "putin", "moscow"],
    "uk": ["united kingdom", "british", "britain"],
    "india": ["india", "indian", "rbi", "modi"],
    "turkey": ["turkey", "turkish", "ankara", "erdogan"],
    "japan": ["japan", "japanese", "boj"],
    "canada": ["canada", "canadian"],
    "eu": ["eurozone", "ecb", "european union"],
    "germany": ["germany", "german"],
    "brazil": ["brazil", "brazilian"],
    "mexico": ["mexico", "mexican"],
    "south korea": ["south korea", "korean"],
    "argentina": ["argentina", "argentine"],
}

MACRO_ENTITIES = {
    "inflation": ["inflation", "cpi", "consumer price"],
    "interest_rates": ["fed rate", "interest rate", "rate cut", "rate hike",
                       "fomc", "federal reserve", "basis points", "bps"],
    "gdp": ["gdp", "economic growth", "gross domestic"],
    "unemployment": ["unemployment", "jobless", "labor market"],
    "recession": ["recession", "economic downturn", "contraction"],
    "oil": ["oil", "crude", "wti", "brent", "petroleum", "opec",
            "strait of hormuz", "energy prices", "gasoline"],
    "crypto": ["bitcoin", "btc", "ethereum", "eth", "solana",
               "crypto", "token", "defi"],
    "stocks": ["s&p 500", "spy", "nasdaq", "qqq", "dow jones",
               "stock market", "earnings"],
    "gold": ["gold price", "gold card", "spot gold", "xau"],
    "bonds": ["treasury", "bond yield", "10-year", "30-year"],
}

POLITICAL_ENTITIES = {
    "trump": ["trump", "donald trump"],
    "fed_chair": ["fed chair", "powell", "warsh", "waller"],
    "war": ["war", "military action", "strike", "ceasefire", "conflict",
            "invasion", "troops", "forces enter"],
    "elections": ["election", "ballot", "primary", "governor",
                  "senate", "congress", "parliament"],
    "tariffs": ["tariff", "trade war", "import duty"],
}

EVENT_TYPES = {
    "data_release": ["will annual", "will monthly", "will quarterly",
                     "gdp growth", "unemployment rate", "inflation increase"],
    "central_bank": ["fed ", "fomc", "ecb", "boj", "rbi", "rate cut",
                     "rate hike", "interest rate"],
    "geopolitical": ["ceasefire", "military action", "invasion",
                     "sanctions", "troops enter", "forces enter"],
    "price_target": ["hit", "reach", "above", "below", "close at",
                     "up or down"],
    "sports": ["win", "playoff", "championship", "seed", "division",
               "regular season", "tournament"],
    "weather": ["temperature", "weather", "highest temp"],
}

# Exclusion lists — prevent false matches
EXCLUSIONS = {
    "gold": ["golden", "gold card", "marigold"],
    "turkey": ["turkey day", "thanksgiving"],
    "fed ": ["fedex", "federer", "federal bureau", "federation"],
    "strike": ["bowling strike", "strike out", "strikeout"],
    "hit": ["hit song", "hit single", "chart hit"],
    "spy": ["spy movie", "spy film", "spy thriller"],
    "token": ["token gesture"],
}

# Resolution period patterns
PERIOD_PATTERNS = [
    (r'in\s+(march|april|may|june|july|august|september|october|november|december)\s*(\d{4})?', 'month'),
    (r'in\s+(q[1-4])\s*(\d{4})?', 'quarter'),
    (r'in\s+(\d{4})', 'year'),
    (r'(?:by|before)\s+(?:end\s+of\s+)?(\d{4})', 'by_year'),
    (r'(?:by|before)\s+(march|april|may|june|july|august|september|october|november|december)\s*(\d{0,4})', 'by_month'),
    (r'after the\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s*(\d{4})?\s*meeting', 'meeting'),
    (r'on\s+(april|may|june)\s+(\d{1,2})', 'specific_date'),
    (r'week of\s+(april|may|june)\s+(\d{1,2})', 'week'),
]

# Metric type patterns — what is being measured
METRIC_PATTERNS = [
    (r'(?:annual|yearly|year.over.year|yoy)\s+inflation', 'annual_inflation'),
    (r'(?:monthly|month.over.month|mom)\s+inflation', 'monthly_inflation'),
    (r'inflation\s+(?:reach|hit|exceed|more than|above)', 'peak_inflation'),
    (r'(?:annual|yearly)\s+(?:cpi|inflation)\s+(?:increase|change)', 'annual_inflation'),
    (r'(?:monthly)\s+(?:cpi|inflation)\s+(?:increase|change)', 'monthly_inflation'),
    (r'(?:gdp|economic)\s+growth', 'gdp_growth'),
    (r'unemployment\s+(?:rate|reach)', 'unemployment'),
    (r'fed\s+(?:rate|interest|funds)', 'fed_rate'),
    (r'(?:oil|crude|wti|brent)\s+(?:price|hit|reach)', 'oil_price'),
]

# Causal link definitions
CAUSAL_LINKS = [
    ("war", "oil", "positive"),
    ("oil", "inflation", "positive"),
    ("inflation", "interest_rates", "positive"),
    ("interest_rates", "stocks", "negative"),
    ("interest_rates", "crypto", "negative"),
    ("interest_rates", "gold", "negative"),
    ("inflation", "recession", "positive"),
    ("recession", "stocks", "negative"),
    ("recession", "unemployment", "positive"),
    ("war", "recession", "positive"),
    ("oil", "gdp", "negative"),
    ("tariffs", "inflation", "positive"),
    ("tariffs", "gdp", "negative"),
]

# Threshold extraction patterns
THRESHOLD_PATTERNS = [
    (r'(?:more than|above|at least|≥|>=|exceed)\s*([\d.]+)\s*%', 'gte_pct'),
    (r'(?:less than|below|under|≤|<=)\s*([\d.]+)\s*%', 'lte_pct'),
    (r'(?:between)\s*([\d.]+)\s*%?\s*(?:and|to|-)\s*([\d.]+)\s*%', 'range_pct'),
    (r'(?:more than|above|hit|reach)\s*\$?([\d,]+)', 'gte_val'),
    (r'(?:less than|below|under)\s*\$?([\d,]+)', 'lte_val'),
    (r'(\d+)\s*(?:bps|basis points)', 'bps'),
    (r'increase by\s*(?:≥)?\s*([\d.]+)\s*%', 'gte_pct'),
]

# Build matchers
country_matcher = EntityMatcher(COUNTRIES)
macro_matcher = EntityMatcher(MACRO_ENTITIES, EXCLUSIONS)
political_matcher = EntityMatcher(POLITICAL_ENTITIES, EXCLUSIONS)
event_matcher = EntityMatcher(EVENT_TYPES, EXCLUSIONS)


# ──────────────────────────────────────────────────────────────
# LAYER 1: PARSER
# ──────────────────────────────────────────────────────────────

def extract_resolution_period(question):
    """Extract when this market resolves and what timeframe it measures."""
    q = question.lower()
    periods = []

    for pattern, ptype in PERIOD_PATTERNS:
        matches = re.findall(pattern, q)
        for match in matches:
            if isinstance(match, tuple):
                periods.append({"type": ptype, "values": [v for v in match if v]})
            else:
                periods.append({"type": ptype, "value": match})

    return periods


def extract_metric_type(question):
    """Determine what is being measured — annual inflation vs peak inflation etc."""
    q = question.lower()
    for pattern, mtype in METRIC_PATTERNS:
        if re.search(pattern, q):
            return mtype
    return None


def parse_market(question, slug="", yes_price=0, volume=0, end_date="", **kwargs):
    """Parse a market question into structured data."""
    parsed = {
        "question": question,
        "slug": slug,
        "yes_price": yes_price,
        "volume": volume,
        "end_date": end_date,
        "countries": country_matcher.match(question),
        "macro_entities": macro_matcher.match(question),
        "political_entities": political_matcher.match(question),
        "event_type": None,
        "metric_type": extract_metric_type(question),
        "resolution_periods": extract_resolution_period(question),
        "thresholds": [],
        "causal_tags": set(),
    }

    # Event type
    event_matches = event_matcher.match(question)
    if event_matches:
        parsed["event_type"] = event_matches[0]

    # Causal tags
    parsed["causal_tags"] = set(parsed["macro_entities"] + parsed["political_entities"])

    # Thresholds
    q = question.lower()
    for pattern, ttype in THRESHOLD_PATTERNS:
        matches = re.findall(pattern, q, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                parsed["thresholds"].append({"type": ttype, "values": list(match)})
            else:
                parsed["thresholds"].append({"type": ttype, "value": match})

    parsed["causal_tags"] = list(parsed["causal_tags"])
    return parsed


# ──────────────────────────────────────────────────────────────
# LAYER 2: CLUSTER & DETECT
# ──────────────────────────────────────────────────────────────

def build_clusters(parsed_markets):
    """Group markets into correlated clusters."""
    clusters = defaultdict(list)
    for pm in parsed_markets:
        for entity in pm["macro_entities"]:
            clusters[f"macro:{entity}"].append(pm)
        for country in pm["countries"]:
            clusters[f"country:{country}"].append(pm)
        for entity in pm["political_entities"]:
            clusters[f"political:{entity}"].append(pm)
        if pm["event_type"]:
            clusters[f"event:{pm['event_type']}"].append(pm)
    return dict(clusters)


def find_causal_chains(clusters):
    """Find markets connected by causal links."""
    chains = []
    for cause, effect, direction in CAUSAL_LINKS:
        cause_key = f"macro:{cause}"
        effect_key = f"macro:{effect}"
        if cause_key not in clusters:
            cause_key = f"political:{cause}"
        if effect_key not in clusters:
            effect_key = f"political:{effect}"

        cause_markets = clusters.get(cause_key, [])
        effect_markets = clusters.get(effect_key, [])

        if cause_markets and effect_markets:
            chains.append({
                "cause": cause, "effect": effect, "direction": direction,
                "cause_markets": len(cause_markets),
                "effect_markets": len(effect_markets),
                "cause_examples": [m["question"][:60] for m in
                    sorted(cause_markets, key=lambda m: m["volume"], reverse=True)[:3]],
                "effect_examples": [m["question"][:60] for m in
                    sorted(effect_markets, key=lambda m: m["volume"], reverse=True)[:3]],
            })
    return chains


def find_deterministic_links(parsed_markets):
    """Find markets with deterministic relationships, matching on metric + period."""
    signals = []

    # Group by metric_type + resolution period for apples-to-apples comparison
    groups = defaultdict(list)
    for pm in parsed_markets:
        metric = pm.get("metric_type")
        if not metric or not pm["thresholds"]:
            continue

        # Build a period key from resolution periods
        periods = pm.get("resolution_periods", [])
        period_key = ""
        for p in periods:
            vals = p.get("values", [p.get("value", "")])
            period_key += "_".join(str(v) for v in vals) + "|"

        if not period_key:
            # Use end_date as fallback period key
            period_key = pm.get("end_date", "unknown")

        key = f"{metric}|{period_key}"
        groups[key].append(pm)

    for key, markets in groups.items():
        if len(markets) < 2:
            continue

        # Collect gte thresholds
        gte_markets = []
        for m in markets:
            for t in m["thresholds"]:
                if t.get("type") == "gte_pct":
                    try:
                        val = float(t.get("value", 0))
                        gte_markets.append((val, m))
                    except ValueError:
                        pass

        gte_markets.sort(key=lambda x: x[0])

        # P(X >= a) should be >= P(X >= b) when a < b
        for i in range(len(gte_markets)):
            for j in range(i + 1, len(gte_markets)):
                low_thresh, low_m = gte_markets[i]
                high_thresh, high_m = gte_markets[j]

                if low_thresh < high_thresh:
                    if low_m["yes_price"] < high_m["yes_price"] - 0.02:
                        edge = high_m["yes_price"] - low_m["yes_price"]
                        signals.append({
                            "type": "DETERMINISTIC_ARB",
                            "group_key": key,
                            "description": (
                                f"P(>={low_thresh}%) = {low_m['yes_price']:.0%} "
                                f"< P(>={high_thresh}%) = {high_m['yes_price']:.0%}"
                            ),
                            "low_market": low_m["question"][:60],
                            "high_market": high_m["question"][:60],
                            "low_price": low_m["yes_price"],
                            "high_price": high_m["yes_price"],
                            "edge": edge,
                            "low_volume": low_m["volume"],
                            "high_volume": high_m["volume"],
                        })

    return signals


# ──────────────────────────────────────────────────────────────
# COMMANDS
# ──────────────────────────────────────────────────────────────

def fetch_all_markets():
    """Fetch all active markets from Gamma API."""
    all_markets = []
    seen = set()
    tags = [
        "politics", "sports", "crypto", "finance", "tech", "culture",
        "geopolitics", "economy", "elections", "entertainment", "science",
        "business", "weather", "iran", "trump",
    ]
    print(f"  Fetching markets from Polymarket...")
    for tag in tags:
        for offset in range(0, 500, 100):
            try:
                r = requests.get(f"{GAMMA_API}/events", params={
                    "tag_slug": tag, "limit": "100", "offset": str(offset),
                    "active": "true", "closed": "false",
                    "order": "volume", "ascending": "false",
                }, timeout=10)
                batch = r.json()
                if not batch:
                    break
                for event in batch:
                    for m in event.get("markets", []):
                        cid = m.get("conditionId", "")
                        if cid in seen:
                            continue
                        seen.add(cid)
                        try:
                            prices = json.loads(m.get("outcomePrices", "[0,0]"))
                            all_markets.append({
                                "question": m.get("question", m.get("groupItemTitle", "")),
                                "slug": event.get("slug", ""),
                                "yes_price": float(prices[0]),
                                "volume": float(m.get("volume", 0)),
                                "end_date": (m.get("endDate") or "")[:10],
                                "condition_id": cid,
                            })
                        except (json.JSONDecodeError, ValueError):
                            continue
                if len(batch) < 100:
                    break
                time.sleep(0.2)
            except Exception:
                break
    print(f"  Fetched {len(all_markets)} markets")
    return all_markets


def cmd_scan(args):
    """Full scan: parse, cluster, find signals."""
    print(f"\n{'='*110}")
    print(f"CORRELATION ENGINE — Market Scan")
    print(f"{'='*110}")

    markets = fetch_all_markets()
    min_vol = getattr(args, 'min_volume', 1000)
    markets = [m for m in markets if m["volume"] >= min_vol]
    print(f"  {len(markets)} markets above ${min_vol:,.0f} volume")

    parsed = [parse_market(**m) for m in markets]
    tagged = [p for p in parsed if p["causal_tags"]]
    print(f"  {len(tagged)} markets with extractable entities")

    clusters = build_clusters(tagged)
    print(f"  {len(clusters)} clusters found")

    # Top clusters
    print(f"\n{'─'*110}")
    print(f"  TOP CLUSTERS (by market count)")
    print(f"{'─'*110}")
    sorted_clusters = sorted(clusters.items(), key=lambda c: len(c[1]), reverse=True)
    for name, members in sorted_clusters[:20]:
        prices = [m["yes_price"] for m in members if 0.01 < m["yes_price"] < 0.99]
        avg_price = sum(prices) / len(prices) if prices else 0
        total_vol = sum(m["volume"] for m in members)
        print(f"  {name:<30s} {len(members):>4d} markets  "
              f"avg YES={avg_price:.0%}  vol=${total_vol:>12,.0f}")

    # Causal chains
    chains = find_causal_chains(clusters)
    if chains:
        print(f"\n{'─'*110}")
        print(f"  CAUSAL CHAINS ({len(chains)} links found)")
        print(f"{'─'*110}")
        for c in chains:
            arrow = "↑" if c["direction"] == "positive" else "↓"
            print(f"\n  {c['cause'].upper()} {arrow}→ {c['effect'].upper()} "
                  f"({c['cause_markets']} × {c['effect_markets']} markets)")
            print(f"    Cause:  {' | '.join(c['cause_examples'][:2])}")
            print(f"    Effect: {' | '.join(c['effect_examples'][:2])}")

    # Deterministic arbs
    det_signals = find_deterministic_links(parsed)
    if det_signals:
        print(f"\n{'─'*110}")
        print(f"  🔴 DETERMINISTIC ARBITRAGE SIGNALS ({len(det_signals)})")
        print(f"{'─'*110}")
        for s in det_signals:
            print(f"\n  {s['description']}")
            print(f"    Group: {s['group_key'][:60]}")
            print(f"    Low:   {s['low_market']}  (vol=${s['low_volume']:,.0f})")
            print(f"    High:  {s['high_market']}  (vol=${s['high_volume']:,.0f})")
            print(f"    Edge:  {s['edge']:.0%}")
    else:
        print(f"\n  No deterministic arbs found (all same-period markets are consistent)")

    # Save
    os.makedirs("data", exist_ok=True)
    with open("data/correlation_scan.json", "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "markets_scanned": len(markets),
            "markets_tagged": len(tagged),
            "clusters": len(clusters),
            "chains": chains,
            "signals": det_signals,
        }, f, indent=2, default=str)
    print(f"\n  Saved: data/correlation_scan.json")
    print(f"{'='*110}")


def cmd_clusters(args):
    """Show markets in a specific cluster."""
    markets = fetch_all_markets()
    min_vol = getattr(args, 'min_volume', 1000)
    markets = [m for m in markets if m["volume"] >= min_vol]
    parsed = [parse_market(**m) for m in markets]
    tagged = [p for p in parsed if p["causal_tags"]]
    clusters = build_clusters(tagged)

    grep = args.grep.lower()
    matching = {k: v for k, v in clusters.items() if grep in k.lower()}
    if not matching:
        for k, v in clusters.items():
            for m in v:
                if grep in m["question"].lower():
                    if k not in matching:
                        matching[k] = v
                    break

    print(f"\n{'='*110}")
    print(f"  CLUSTERS matching '{args.grep}' — {len(matching)} found")
    print(f"{'='*110}")
    for name, members in sorted(matching.items(), key=lambda c: len(c[1]), reverse=True):
        print(f"\n  {name} ({len(members)} markets)")
        print(f"  {'─'*100}")
        for m in sorted(members, key=lambda m: m["volume"], reverse=True)[:15]:
            q = m["question"][:65]
            metric = m.get("metric_type", "")
            periods = m.get("resolution_periods", [])
            period_str = ""
            if periods:
                p = periods[0]
                vals = p.get("values", [p.get("value", "")])
                period_str = " ".join(str(v) for v in vals)
            print(f"    {m['yes_price']:>4.0%}  ${m['volume']:>10,.0f}  "
                  f"{metric or '':<20s} {period_str:<12s} {q}")


def cmd_chain(args):
    """Show the full causal chain for a topic."""
    topic = args.topic.lower()
    visited = {topic}
    queue = [topic]
    chain = []
    while queue:
        current = queue.pop(0)
        for c, e, d in CAUSAL_LINKS:
            if c == current and e not in visited:
                chain.append((c, e, d))
                visited.add(e)
                queue.append(e)
            elif e == current and c not in visited:
                chain.append((c, e, d))
                visited.add(c)
                queue.append(c)

    print(f"\n{'='*80}")
    print(f"  CAUSAL CHAIN: {topic.upper()}")
    print(f"{'='*80}")
    for c, e, d in chain:
        arrow = "↑" if d == "positive" else "↓"
        print(f"  {c.upper():<20s} ──{arrow}──→  {e.upper()}")

    print(f"\n  Fetching market prices along the chain...")
    markets = fetch_all_markets()
    markets = [m for m in markets if m["volume"] >= 1000]
    parsed = [parse_market(**m) for m in markets]

    for c, e, d in chain:
        related = [p for p in parsed if c in p["causal_tags"] or e in p["causal_tags"]]
        if related:
            print(f"\n  {c.upper()} → {e.upper()} markets:")
            for m in sorted(related, key=lambda m: m["volume"], reverse=True)[:5]:
                print(f"    {m['yes_price']:>4.0%}  ${m['volume']:>10,.0f}  {m['question'][:60]}")


def main():
    parser = argparse.ArgumentParser(description="Correlation Engine")
    parser.add_argument("--min-volume", type=float, default=1000)
    subs = parser.add_subparsers(dest="command", required=True)

    subs.add_parser("scan", help="Full scan")
    p_c = subs.add_parser("clusters", help="Show cluster contents")
    p_c.add_argument("--grep", type=str, required=True)
    p_ch = subs.add_parser("chain", help="Show causal chain")
    p_ch.add_argument("--topic", type=str, required=True)

    args = parser.parse_args()
    {"scan": cmd_scan, "clusters": cmd_clusters, "chain": cmd_chain}[args.command](args)


if __name__ == "__main__":
    main()
