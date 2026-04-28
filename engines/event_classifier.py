"""
engines/event_classifier.py — LLM-Based Event Type Classifier

Classifies Polymarket market titles into event categories using an LLM
(DeepSeek or OpenAI-compatible). Maintains a growing taxonomy of past
classifications for few-shot context and consistency.

Categories:
    geopolitical  — wars, military, diplomacy, sanctions, territory
    economic      — inflation, GDP, employment, central bank policy
    financial     — crypto prices, stocks, IPOs, trading, tokens
    political     — elections, legislation, appointments, impeachment
    legal         — trials, sentencing, lawsuits, rulings
    sports        — games, championships, leagues, player performance
    esports       — competitive gaming (CS, Dota, LoL, Valorant)
    tech          — product releases, acquisitions, AI developments
    pop_culture   — celebrities, music, movies, social media
    weather       — temperature, storms, natural events
    health        — FDA, outbreaks, vaccines, drug approvals
    other         — doesn't fit any category

Usage:
    python -m engines.event_classifier classify "Will Bitcoin hit $100k?"
    python -m engines.event_classifier reclassify                        # Reclassify all spike scanner markets
    python -m engines.event_classifier taxonomy                          # Show classification stats
    python -m engines.event_classifier correct --slug some-slug --type financial
"""
import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

DB_PATH = Path("data/spike_scanner.db")

VALID_TYPES = [
    "geopolitical", "economic", "financial", "political", "legal",
    "sports", "esports", "tech", "pop_culture", "weather", "health", "other"
]

SYSTEM_PROMPT = """You are an event classifier for prediction markets. Given a market title/question, classify it into exactly ONE of these categories:

- geopolitical: wars, military action, diplomacy, sanctions, territorial disputes, ceasefire, NATO, invasions
- economic: inflation, CPI, GDP, unemployment, Fed/central bank policy, interest rates, tariffs, recession
- financial: crypto prices (Bitcoin, Ethereum), stock prices, IPOs, tokens, airdrops, market cap, trading
- political: elections, presidential races, legislation, appointments, impeachment, endorsements, political parties
- legal: criminal trials, sentencing, lawsuits, court rulings, convictions, extradition
- sports: traditional sports - NBA, NFL, NHL, MLB, soccer/football, tennis, golf, MMA, boxing, wrestling
- esports: competitive video gaming - Counter-Strike, Dota 2, League of Legends, Valorant, esports tournaments
- tech: product launches, software releases (GPT, iOS), company acquisitions, AI developments, app launches
- pop_culture: celebrities, music albums, movies, TV shows, social media personalities, YouTube, TikTok
- weather: temperature predictions, hurricanes, earthquakes, floods, natural disasters, climate
- health: FDA approvals, drug trials, disease outbreaks, pandemics, vaccines, WHO declarations
- other: anything that doesn't clearly fit the above categories

IMPORTANT DISTINCTIONS:
- "Counter-Strike" is esports, NOT geopolitical (despite containing "strike")
- "Elon Musk tweets" is pop_culture, NOT geopolitical or tech
- Temperature/weather predictions are weather, NOT tech
- Crypto price predictions are financial, NOT tech
- Soccer/football relegation/league tables are sports, NOT political
- "Will X win" in gaming context is esports, NOT sports

Respond with ONLY a JSON object, no markdown, no explanation:
{"category": "<category>", "confidence": <0.0-1.0>, "reasoning": "<brief one-line reason>"}"""


def get_llm_client():
    """Get an OpenAI-compatible client for DeepSeek or fallback."""
    try:
        from openai import OpenAI
    except ImportError:
        print("  ❌ pip install openai required")
        sys.exit(1)

    # Try DeepSeek first (cheapest)
    ds_key = os.getenv("DEEPSEEK_API_KEY")
    if ds_key:
        return OpenAI(api_key=ds_key, base_url="https://api.deepseek.com"), "deepseek-chat"

    # Fallback to OpenAI
    oai_key = os.getenv("OPENAI_API_KEY")
    if oai_key:
        return OpenAI(api_key=oai_key), "gpt-4o-mini"

    # Fallback to Groq
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        return OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1"), "llama-3.1-8b-instant"

    print("  ❌ No API key found. Set DEEPSEEK_API_KEY, OPENAI_API_KEY, or GROQ_API_KEY in .env")
    sys.exit(1)


def init_taxonomy_table(conn):
    """Add taxonomy table to spike scanner DB if not exists."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS taxonomy (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slug TEXT UNIQUE,
            question TEXT,
            classified_as TEXT,
            confidence REAL,
            reasoning TEXT,
            corrected_to TEXT,
            provider TEXT,
            classified_at TEXT
        )
    """)
    conn.commit()


def get_few_shot_examples(conn, n=10):
    """Get recent correct classifications for few-shot context."""
    # Prefer corrected examples (human-verified), then high-confidence auto
    examples = conn.execute("""
        SELECT question, COALESCE(corrected_to, classified_as) as final_type, reasoning
        FROM taxonomy
        WHERE classified_as IS NOT NULL
        ORDER BY
            CASE WHEN corrected_to IS NOT NULL THEN 0 ELSE 1 END,
            confidence DESC
        LIMIT ?
    """, (n,)).fetchall()

    if not examples:
        return ""

    lines = []
    for q, t, r in examples:
        lines.append(f'  "{q[:80]}" → {t}')

    return "\n\nRecent classifications for consistency:\n" + "\n".join(lines)


def classify_single(client, model, question, few_shot=""):
    """Classify a single market title via LLM."""
    prompt = f"{few_shot}\n\nClassify this market:\n\"{question}\""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=150,
        )

        text = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        text = text.replace("```json", "").replace("```", "").strip()

        result = json.loads(text)
        category = result.get("category", "other").lower()
        confidence = float(result.get("confidence", 0.5))
        reasoning = result.get("reasoning", "")

        # Validate category
        if category not in VALID_TYPES:
            category = "other"

        return category, confidence, reasoning

    except json.JSONDecodeError:
        # Try to extract category from raw text
        text_lower = text.lower() if text else ""
        for vt in VALID_TYPES:
            if vt in text_lower:
                return vt, 0.3, "parsed from raw text"
        return "other", 0.1, f"JSON parse failed: {text[:50]}"

    except Exception as e:
        return "other", 0.0, f"API error: {str(e)[:50]}"


def classify_batch(questions, batch_size=20):
    """Classify a batch of market titles.

    Args:
        questions: list of (slug, question) tuples
        batch_size: how many to classify per API call (1 for now, batch later)

    Returns:
        list of (slug, question, category, confidence, reasoning) tuples
    """
    client, model = get_llm_client()
    conn = sqlite3.connect(str(DB_PATH))
    init_taxonomy_table(conn)

    few_shot = get_few_shot_examples(conn)
    results = []
    last_progress = time.time()
    start = time.time()

    print(f"\n  Classifying {len(questions)} markets via {model}...")
    if few_shot:
        print(f"  Using {few_shot.count('→')} few-shot examples for consistency")

    for i, (slug, question) in enumerate(questions):
        # Skip if already classified and not corrected
        existing = conn.execute(
            "SELECT classified_as FROM taxonomy WHERE slug=?", (slug,)).fetchone()
        if existing:
            results.append((slug, question, existing[0], 1.0, "cached"))
            continue

        category, confidence, reasoning = classify_single(
            client, model, question, few_shot)

        # Store in taxonomy
        conn.execute("""
            INSERT OR REPLACE INTO taxonomy
            (slug, question, classified_as, confidence, reasoning, provider, classified_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (slug, question[:200], category, confidence, reasoning,
              model, datetime.now(timezone.utc).isoformat()))

        # Also update the markets table
        conn.execute(
            "UPDATE markets SET event_type=?, event_confidence=? WHERE slug=?",
            (category, confidence, slug))

        conn.commit()
        results.append((slug, question, category, confidence, reasoning))

        # Progress every 30 seconds
        now = time.time()
        if now - last_progress >= 30 or i == len(questions) - 1:
            elapsed = now - start
            pct = (i + 1) / len(questions) * 100
            remaining = (elapsed / (i + 1)) * (len(questions) - i - 1) if i > 0 else 0
            api_calls = sum(1 for r in results if r[4] != "cached")
            print(f"    [{datetime.now().strftime('%H:%M:%S')}] "
                  f"{i+1}/{len(questions)} ({pct:.0f}%) | "
                  f"{api_calls} API calls | ~{remaining:.0f}s remaining")
            last_progress = now

        # Rate limiting — DeepSeek is generous but be polite
        time.sleep(0.3)

        # Refresh few-shot every 50 classifications
        if (i + 1) % 50 == 0:
            few_shot = get_few_shot_examples(conn)

    conn.close()
    return results


# ═══════════════════════════════════════════════════════
# COMMANDS
# ═══════════════════════════════════════════════════════

def cmd_classify(args):
    """Classify a single market title."""
    question = args.question
    client, model = get_llm_client()

    conn = sqlite3.connect(str(DB_PATH))
    init_taxonomy_table(conn)
    few_shot = get_few_shot_examples(conn)
    conn.close()

    print(f"\n  Classifying: \"{question}\"")
    print(f"  Model: {model}")

    category, confidence, reasoning = classify_single(
        client, model, question, few_shot)

    print(f"\n  Category:   {category}")
    print(f"  Confidence: {confidence:.0%}")
    print(f"  Reasoning:  {reasoning}")


def cmd_reclassify(args):
    """Reclassify all markets in the spike scanner database."""
    if not DB_PATH.exists():
        print(f"  ❌ No database at {DB_PATH}. Run spike_scanner collect first.")
        return

    conn = sqlite3.connect(str(DB_PATH))
    init_taxonomy_table(conn)

    # Get all markets
    markets = conn.execute("""
        SELECT slug, question FROM markets
        WHERE question IS NOT NULL AND question != ''
        ORDER BY volume DESC
    """).fetchall()

    # Check how many already classified
    already = conn.execute("SELECT COUNT(*) FROM taxonomy").fetchone()[0]
    force = getattr(args, "force", False)

    if force:
        # Clear existing classifications
        conn.execute("DELETE FROM taxonomy")
        conn.commit()
        already = 0

    print(f"\n{'='*80}")
    print(f"  EVENT CLASSIFIER — Reclassify All Markets")
    print(f"  Markets in DB: {len(markets)}")
    print(f"  Already classified: {already}")
    print(f"  Force reclassify: {force}")
    print(f"{'='*80}")

    questions = [(m[0], m[1]) for m in markets]
    results = classify_batch(questions)

    # Summary
    type_counts = {}
    for _, _, cat, _, _ in results:
        type_counts[cat] = type_counts.get(cat, 0) + 1

    print(f"\n  Classification Results:")
    print(f"  {'Type':<15s} {'Count':>6s}")
    print(f"  {'─'*25}")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t:<15s} {c:>6d}")

    # Show a few examples per type
    print(f"\n  Sample Classifications:")
    for t in sorted(type_counts.keys()):
        examples = [(q, r) for _, q, c, _, r in results if c == t][:3]
        print(f"\n  [{t}]")
        for q, r in examples:
            print(f"    {q[:70]}")
            if r and r != "cached":
                print(f"      → {r}")

    conn.close()
    print(f"\n{'='*80}")


def cmd_taxonomy(args):
    """Show taxonomy statistics."""
    if not DB_PATH.exists():
        print(f"  ❌ No database at {DB_PATH}")
        return

    conn = sqlite3.connect(str(DB_PATH))
    init_taxonomy_table(conn)

    total = conn.execute("SELECT COUNT(*) FROM taxonomy").fetchone()[0]
    corrected = conn.execute(
        "SELECT COUNT(*) FROM taxonomy WHERE corrected_to IS NOT NULL").fetchone()[0]

    print(f"\n{'='*80}")
    print(f"  EVENT TAXONOMY")
    print(f"  Total classified: {total}")
    print(f"  Human corrected:  {corrected}")
    print(f"{'='*80}")

    types = conn.execute("""
        SELECT COALESCE(corrected_to, classified_as) as final_type,
               COUNT(*),
               AVG(confidence)
        FROM taxonomy
        GROUP BY final_type
        ORDER BY COUNT(*) DESC
    """).fetchall()

    print(f"\n  {'Type':<15s} {'Count':>6s} {'AvgConf':>8s}")
    print(f"  {'─'*32}")
    for t in types:
        print(f"  {t[0]:<15s} {t[1]:>6d} {t[2]:>7.0%}")

    # Show misclassifications (corrected ones)
    if corrected > 0:
        print(f"\n  Corrections ({corrected}):")
        corrections = conn.execute("""
            SELECT question, classified_as, corrected_to
            FROM taxonomy WHERE corrected_to IS NOT NULL
            LIMIT 20
        """).fetchall()
        for c in corrections:
            print(f"    {c[0][:50]} : {c[1]} → {c[2]}")

    # Cross-reference with spikes
    spike_types = conn.execute("""
        SELECT t.classified_as, COUNT(s.id), AVG(ABS(s.spike_pct))
        FROM spikes s
        JOIN taxonomy t ON s.slug = t.slug
        GROUP BY t.classified_as
        ORDER BY COUNT(s.id) DESC
    """).fetchall()

    if spike_types:
        print(f"\n  Spikes by Corrected Type:")
        print(f"  {'Type':<15s} {'Spikes':>7s} {'AvgMove':>8s}")
        print(f"  {'─'*32}")
        for t in spike_types:
            print(f"  {t[0]:<15s} {t[1]:>7d} {t[2]:>+7.1f}%")

    conn.close()
    print(f"\n{'='*80}")


def cmd_correct(args):
    """Manually correct a classification."""
    slug = args.slug
    new_type = args.type

    if new_type not in VALID_TYPES:
        print(f"  ❌ Invalid type: {new_type}")
        print(f"  Valid: {', '.join(VALID_TYPES)}")
        return

    conn = sqlite3.connect(str(DB_PATH))
    init_taxonomy_table(conn)

    existing = conn.execute(
        "SELECT question, classified_as FROM taxonomy WHERE slug=?",
        (slug,)).fetchone()

    if not existing:
        print(f"  ❌ Slug not found: {slug}")
        conn.close()
        return

    print(f"  Market:  {existing[0]}")
    print(f"  Was:     {existing[1]}")
    print(f"  Now:     {new_type}")

    conn.execute(
        "UPDATE taxonomy SET corrected_to=? WHERE slug=?",
        (new_type, slug))
    conn.execute(
        "UPDATE markets SET event_type=? WHERE slug=?",
        (new_type, slug))
    conn.commit()
    conn.close()

    print(f"  ✅ Corrected. This will be used as a few-shot example going forward.")


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="LLM Event Classifier")
    subs = parser.add_subparsers(dest="command")

    p_cls = subs.add_parser("classify", help="Classify a single title")
    p_cls.add_argument("question", type=str)

    p_recls = subs.add_parser("reclassify", help="Reclassify all markets")
    p_recls.add_argument("--force", action="store_true",
                         help="Clear existing and reclassify all")

    subs.add_parser("taxonomy", help="Show taxonomy stats")

    p_cor = subs.add_parser("correct", help="Correct a classification")
    p_cor.add_argument("--slug", required=True)
    p_cor.add_argument("--type", required=True, choices=VALID_TYPES)

    args = parser.parse_args()

    if args.command == "classify":
        cmd_classify(args)
    elif args.command == "reclassify":
        cmd_reclassify(args)
    elif args.command == "taxonomy":
        cmd_taxonomy(args)
    elif args.command == "correct":
        cmd_correct(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
