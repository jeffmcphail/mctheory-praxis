"""
engines/spike_scanner.py — Historical Event-Driven Spike Scanner

Phase 1c Step 1: Data Collection for Event-Driven Spike Predictor

Scans resolved and active Polymarket markets for price spikes:
  1. Fetches price history from CLOB API
  2. Detects rapid price movements (>X% in <Y minutes)
  3. Classifies event type from market title/metadata
  4. Stores spike data in SQLite for ML training

Usage:
    python -m engines.spike_scanner collect                       # Collect spike data from recent markets
    python -m engines.spike_scanner collect --days 30             # Look back 30 days
    python -m engines.spike_scanner collect --min-volume 50000    # Only high-volume markets
    python -m engines.spike_scanner analyze                       # Show spike statistics
    python -m engines.spike_scanner taxonomy                      # Show event type breakdown
"""
import argparse
import json
import os
import re
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
DB_PATH = Path("data/spike_scanner.db")

# Spike detection parameters
SPIKE_THRESHOLD_PCT = 5.0    # Minimum price change to qualify as a spike (%)
SPIKE_WINDOW_MINS = 60       # Time window to detect spike (minutes)
MIN_PRICE = 0.05             # Ignore markets below 5¢ (noise)
MAX_PRICE = 0.95             # Ignore markets above 95¢ (already resolved)


# ═══════════════════════════════════════════════════════
# EVENT TAXONOMY
# ═══════════════════════════════════════════════════════

EVENT_TYPES = {
    "geopolitical": [
        "war", "invasion", "invade", "ceasefire", "military", "nato", "nuclear",
        "iran", "russia", "ukraine", "china", "taiwan", "strike", "troops",
        "sanctions", "annex", "sovereignty", "clash", "missile", "attack",
        "hamas", "hezbollah", "israel", "gaza", "syria", "north korea",
    ],
    "economic": [
        "inflation", "cpi", "gdp", "unemployment", "fed", "rate cut", "rate hike",
        "interest rate", "fomc", "recession", "jobs", "payroll", "pce", "tariff",
        "trade deficit", "debt ceiling", "treasury", "yield", "bond",
    ],
    "financial": [
        "bitcoin", "btc", "ethereum", "eth", "crypto", "stock", "s&p",
        "nasdaq", "ipo", "market cap", "fdv", "price", "trading",
        "solana", "token", "airdrop", "defi",
    ],
    "political": [
        "president", "election", "vote", "democrat", "republican", "trump",
        "biden", "congress", "senate", "governor", "impeach", "indict",
        "supreme court", "nominee", "endorse", "legislation", "bill",
    ],
    "legal": [
        "sentenced", "trial", "verdict", "guilty", "lawsuit", "ruling",
        "indictment", "conviction", "prison", "court", "judge", "epstein",
        "weinstein", "extradite",
    ],
    "sports": [
        "win", "championship", "nba", "nfl", "nhl", "mlb", "premier league",
        "la liga", "bundesliga", "serie a", "champions league", "world cup",
        "stanley cup", "super bowl", "playoffs", "seed", "relegat", "masters",
        "goal scorer", "ballon d'or",
    ],
    "tech": [
        "gpt", "openai", "ai", "release", "launch", "apple", "google",
        "microsoft", "meta", "tesla", "spacex", "neuralink", "tiktok",
        "acquire", "merger",
    ],
    "pop_culture": [
        "album", "movie", "celebrity", "pregnant", "married", "divorce",
        "rihanna", "drake", "taylor swift", "kanye", "kardashian",
        "gta vi", "game", "youtube", "mrbeast", "viral",
    ],
    "weather": [
        "temperature", "weather", "hurricane", "earthquake", "flood",
        "wildfire", "tornado", "storm",
    ],
    "health": [
        "covid", "pandemic", "vaccine", "fda", "drug", "approved",
        "outbreak", "bird flu", "who",
    ],
}


def classify_event(title, question=""):
    """Classify an event by type based on title keywords.
    Returns (primary_type, confidence, matched_keywords).
    """
    text = (title + " " + question).lower()
    scores = {}

    for event_type, keywords in EVENT_TYPES.items():
        matched = [kw for kw in keywords if kw in text]
        if matched:
            scores[event_type] = len(matched)

    if not scores:
        return ("unknown", 0, [])

    best = max(scores, key=scores.get)
    total_matches = sum(scores.values())
    confidence = scores[best] / total_matches if total_matches > 0 else 0

    matched_kws = [kw for kw in EVENT_TYPES[best] if kw in text]
    return (best, confidence, matched_kws)


# ═══════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS markets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slug TEXT UNIQUE,
            question TEXT,
            condition_id TEXT,
            event_type TEXT,
            event_confidence REAL,
            event_keywords TEXT,
            volume REAL,
            liquidity REAL,
            neg_risk INTEGER,
            created_at TEXT,
            end_date TEXT,
            resolved INTEGER,
            resolution TEXT,
            yes_token TEXT,
            no_token TEXT,
            tick_size REAL,
            price_history_fetched INTEGER DEFAULT 0,
            num_spikes INTEGER DEFAULT 0,
            scanned_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS price_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slug TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            price REAL NOT NULL,
            UNIQUE(slug, timestamp)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS spikes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slug TEXT NOT NULL,
            question TEXT,
            event_type TEXT,
            spike_start_ts INTEGER,
            spike_peak_ts INTEGER,
            spike_start_price REAL,
            spike_peak_price REAL,
            spike_pct REAL,
            spike_duration_mins REAL,
            direction TEXT,
            reversion_pct REAL,
            reversion_mins REAL,
            volume REAL,
            context TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ph_slug ON price_history(slug)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_spikes_type ON spikes(event_type)")
    conn.commit()
    return conn


# ═══════════════════════════════════════════════════════
# DATA COLLECTION
# ═══════════════════════════════════════════════════════

def fetch_price_history(token_id, fidelity=60):
    """Fetch price history from CLOB API.

    Args:
        token_id: The CLOB token ID
        fidelity: Candle interval in seconds (60=1min, 3600=1hr)

    Returns:
        List of (timestamp, price) tuples sorted by time.
        Tries fidelity=60 first, falls back to 3600 if no data.
    """
    for fid in [fidelity, 3600, 86400]:  # 1min → 1hr → 1day fallback
        try:
            r = requests.get(f"{CLOB_API}/prices-history", params={
                "market": token_id,
                "interval": "all",
                "fidelity": fid,
            }, timeout=30)

            if r.status_code != 200:
                continue

            data = r.json()
            if not data or not isinstance(data, dict):
                continue

            history = data.get("history", [])
            if not history:
                continue

            points = []
            for point in history:
                ts = int(point.get("t", 0))
                price = float(point.get("p", 0))
                if ts > 0 and 0 <= price <= 1:
                    points.append((ts, price))

            if points:
                points.sort(key=lambda x: x[0])
                return points

        except Exception:
            continue

    return []


def detect_spikes(prices, threshold_pct=SPIKE_THRESHOLD_PCT,
                  window_mins=SPIKE_WINDOW_MINS, resolution_ts=None):
    """Detect price spikes in a time series.

    A spike is a price move of >threshold_pct within window_mins.
    If resolution_ts is provided, exclude moves within 2 hours of resolution
    (those are just the market settling, not tradeable spikes).

    Returns list of spike dicts.
    """
    if len(prices) < 5:
        return []

    # Exclude price data within 2 hours of resolution
    RESOLUTION_BUFFER_SECS = 7200  # 2 hours
    if resolution_ts:
        cutoff_ts = resolution_ts - RESOLUTION_BUFFER_SECS
        prices = [(t, p) for t, p in prices if t < cutoff_ts]
        if len(prices) < 5:
            return []

    spikes = []
    window_secs = window_mins * 60

    for i in range(len(prices)):
        t_start, p_start = prices[i]

        # Skip extreme prices (noise near 0 or 1)
        if p_start < MIN_PRICE or p_start > MAX_PRICE:
            continue

        # Look forward within window
        best_move = 0
        best_j = i
        best_peak = p_start

        for j in range(i + 1, len(prices)):
            t_j, p_j = prices[j]

            if t_j - t_start > window_secs:
                break

            pct_move = (p_j - p_start) / p_start * 100
            if abs(pct_move) > abs(best_move):
                best_move = pct_move
                best_j = j
                best_peak = p_j

        if abs(best_move) >= threshold_pct:
            t_peak = prices[best_j][0]
            duration_mins = (t_peak - t_start) / 60

            # Check for reversion after the spike
            reversion_pct = 0
            reversion_mins = 0
            for k in range(best_j + 1, min(best_j + 120, len(prices))):
                t_k, p_k = prices[k]
                rev = (p_k - best_peak) / best_peak * 100 if best_peak > 0 else 0
                # Reversion goes opposite to spike direction
                if best_move > 0 and rev < reversion_pct:
                    reversion_pct = rev
                    reversion_mins = (t_k - t_peak) / 60
                elif best_move < 0 and rev > abs(reversion_pct):
                    reversion_pct = rev
                    reversion_mins = (t_k - t_peak) / 60

            spike = {
                "start_ts": t_start,
                "peak_ts": t_peak,
                "start_price": p_start,
                "peak_price": best_peak,
                "pct": best_move,
                "duration_mins": duration_mins,
                "direction": "UP" if best_move > 0 else "DOWN",
                "reversion_pct": reversion_pct,
                "reversion_mins": reversion_mins,
            }
            spikes.append(spike)

            # Skip ahead past this spike to avoid double-counting
            # (jump to peak + half window)
            skip_to_ts = t_peak + window_secs // 2
            while i < len(prices) - 1 and prices[i + 1][0] < skip_to_ts:
                i += 1

    return spikes


# ═══════════════════════════════════════════════════════
# COMMANDS
# ═══════════════════════════════════════════════════════

def cmd_collect(args):
    """Collect price history and detect spikes from recent markets."""
    days = getattr(args, "days", 14)
    min_volume = getattr(args, "min_volume", 10000)
    max_markets = getattr(args, "max_markets", 200)
    fidelity = getattr(args, "fidelity", 60)  # 1-minute candles

    conn = init_db()
    start_time = time.time()

    print(f"\n{'='*100}")
    print(f"  SPIKE SCANNER — Historical Event Data Collection")
    print(f"  Phase 1c: Building training data for event-driven predictor")
    print(f"  Lookback: {days} days  |  Min volume: ${min_volume:,.0f}  |  Max markets: {max_markets}")
    print(f"  Fidelity: {fidelity}s candles  |  Spike threshold: {SPIKE_THRESHOLD_PCT}%")
    print(f"{'='*100}")

    # Fetch recently resolved markets (best for training — we know the outcome)
    print(f"\n  Fetching resolved markets from Gamma API...")
    print(f"  (Fetching up to 5000 resolved markets sorted by volume...)")

    all_markets = []
    offset = 0
    limit = 100
    fetch_start = time.time()

    while len(all_markets) < 5000:
        try:
            r = requests.get(f"{GAMMA_API}/markets", params={
                "closed": "true",
                "limit": limit,
                "offset": offset,
                "order": "volume",
                "ascending": "false",
            }, timeout=15)
            batch = r.json()
            if not batch:
                break
            all_markets.extend(batch)
            offset += limit

            if len(all_markets) % 500 < limit:
                elapsed = time.time() - fetch_start
                print(f"    {len(all_markets):>5,d} resolved markets fetched... ({elapsed:.0f}s)")

            if len(batch) < limit:
                break
        except Exception as e:
            print(f"    ⚠ Fetch error at offset {offset}: {e}")
            # Don't break — try to continue
            offset += limit
            if offset > 10000:
                break
            continue

    elapsed = time.time() - fetch_start
    print(f"  Fetched {len(all_markets):,d} resolved markets in {elapsed:.0f}s")

    # Filter by volume and recency
    filtered = []
    for m in all_markets:
        vol = float(m.get("volume", 0) or 0)
        if vol < min_volume:
            continue

        token_ids = json.loads(m.get("clobTokenIds", "[]"))
        if len(token_ids) < 1:
            continue

        # Check if already scanned
        slug = m.get("slug", "")
        existing = conn.execute("SELECT price_history_fetched FROM markets WHERE slug=?",
                                (slug,)).fetchone()
        if existing and existing[0]:
            continue

        filtered.append(m)

    filtered.sort(key=lambda m: float(m.get("volume", 0) or 0), reverse=True)
    filtered = filtered[:max_markets]

    print(f"  {len(filtered)} markets to scan (after volume/dedup filter)")
    if not filtered:
        print(f"  Nothing to scan.")
        conn.close()
        return

    est_time = len(filtered) * 1.5  # ~1.5s per market (API call + processing)
    print(f"  Estimated time: ~{est_time:.0f}s ({est_time/60:.1f} min)")

    # Process each market
    total_spikes = 0
    markets_with_spikes = 0
    markets_with_data = 0
    total_price_points = 0
    type_counts = {}
    last_progress = time.time()
    scan_start = time.time()

    for i, m in enumerate(filtered):
        slug = m.get("slug", "")
        question = m.get("question", "")
        token_ids = json.loads(m.get("clobTokenIds", "[]"))
        yes_token = token_ids[0] if token_ids else ""
        no_token = token_ids[1] if len(token_ids) > 1 else ""
        vol = float(m.get("volume", 0) or 0)

        # Classify event type (keyword-based for now, LLM upgrade later)
        event_type, confidence, keywords = classify_event(
            m.get("question", ""), m.get("description", ""))

        type_counts[event_type] = type_counts.get(event_type, 0) + 1

        # Parse resolution timestamp for spike filtering
        resolution_ts = None
        end_date = m.get("endDate", "") or m.get("end_date_iso", "")
        if end_date:
            try:
                # Handle various date formats
                for fmt in ["%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ",
                            "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"]:
                    try:
                        dt = datetime.strptime(end_date, fmt).replace(tzinfo=timezone.utc)
                        resolution_ts = int(dt.timestamp())
                        break
                    except ValueError:
                        continue
            except Exception:
                pass

        # Fetch price history (with fidelity fallback)
        prices = fetch_price_history(yes_token, fidelity=fidelity)

        if prices:
            markets_with_data += 1
            total_price_points += len(prices)

        # Store market metadata
        conn.execute("""
            INSERT OR REPLACE INTO markets
            (slug, question, condition_id, event_type, event_confidence,
             event_keywords, volume, liquidity, neg_risk, end_date,
             resolved, yes_token, no_token, tick_size,
             price_history_fetched, scanned_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            slug, question, m.get("conditionId", ""),
            event_type, confidence, json.dumps(keywords),
            vol, float(m.get("liquidityClob", 0) or 0),
            1 if m.get("negRisk") else 0,
            end_date,
            1, yes_token[:40], no_token[:40],
            float(m.get("orderPriceMinTickSize", 0.01) or 0.01),
            1 if prices else 0,
            datetime.now(timezone.utc).isoformat(),
        ))

        # Store price history
        if prices:
            for ts, price in prices:
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO price_history (slug, timestamp, price) VALUES (?, ?, ?)",
                        (slug, ts, price))
                except Exception:
                    pass

            # Detect spikes (with resolution filter — exclude final 2 hours)
            spikes = detect_spikes(prices, resolution_ts=resolution_ts)

            if spikes:
                markets_with_spikes += 1
                total_spikes += len(spikes)

                # Update market record
                conn.execute("UPDATE markets SET num_spikes=? WHERE slug=?",
                             (len(spikes), slug))

                for spike in spikes:
                    conn.execute("""
                        INSERT INTO spikes
                        (slug, question, event_type, spike_start_ts, spike_peak_ts,
                         spike_start_price, spike_peak_price, spike_pct,
                         spike_duration_mins, direction, reversion_pct,
                         reversion_mins, volume, context)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        slug, question[:100], event_type,
                        spike["start_ts"], spike["peak_ts"],
                        spike["start_price"], spike["peak_price"],
                        spike["pct"], spike["duration_mins"],
                        spike["direction"], spike["reversion_pct"],
                        spike["reversion_mins"], vol,
                        json.dumps(keywords),
                    ))

        conn.commit()

        # Progress every 30 seconds
        now = time.time()
        if now - last_progress >= 30:
            elapsed = now - scan_start
            pct = (i + 1) / len(filtered) * 100
            remaining = (elapsed / (i + 1)) * (len(filtered) - i - 1)
            data_pct = markets_with_data / (i + 1) * 100 if i > 0 else 0
            print(f"    [{datetime.now().strftime('%H:%M:%S')}] "
                  f"{i+1}/{len(filtered)} ({pct:.0f}%) | "
                  f"{markets_with_data} w/data ({data_pct:.0f}%), "
                  f"{total_spikes} spikes in {markets_with_spikes} mkts | "
                  f"~{remaining:.0f}s remaining")
            last_progress = now

        time.sleep(0.3)  # Rate limit

    conn.commit()
    elapsed = time.time() - start_time

    # Summary
    print(f"\n{'─'*100}")
    print(f"  COLLECTION COMPLETE")
    print(f"{'─'*100}")
    print(f"  Markets scanned:         {len(filtered)}")
    print(f"  Markets with price data: {markets_with_data} ({markets_with_data/len(filtered)*100:.0f}%)")
    print(f"  Total price datapoints:  {total_price_points:,}")
    print(f"  Markets with spikes:     {markets_with_spikes}")
    print(f"  Total spikes detected:   {total_spikes}")
    print(f"  Resolution filter:       excluding moves within 2h of close")
    print(f"  Scan duration:           {elapsed:.0f}s")

    print(f"\n  Event Type Distribution:")
    for etype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        spike_count = conn.execute(
            "SELECT COUNT(*) FROM spikes WHERE event_type=?", (etype,)).fetchone()[0]
        print(f"    {etype:<15s} {count:>4d} markets, {spike_count:>4d} spikes")

    if total_spikes > 0:
        print(f"\n  Top 10 Largest Spikes:")
        print(f"  {'#':<4s} {'Market':<50s} {'Type':<12s} {'Move':>8s} {'Dur':>6s} "
              f"{'Revert':>8s} {'Dir':<5s} {'Vol':>12s}")
        print(f"  {'─'*110}")

        top_spikes = conn.execute("""
            SELECT question, event_type, spike_pct, spike_duration_mins,
                   reversion_pct, direction, volume,
                   spike_start_price, spike_peak_price
            FROM spikes ORDER BY ABS(spike_pct) DESC LIMIT 10
        """).fetchall()

        for j, s in enumerate(top_spikes, 1):
            q = s[0][:49]
            print(f"  {j:<4d} {q:<50s} {s[1]:<12s} {s[2]:>+7.1f}% {s[3]:>5.0f}m "
                  f"{s[4]:>+7.1f}% {s[5]:<5s} ${s[6]:>10,.0f}")

    print(f"\n  Data saved to: {DB_PATH}")
    print(f"{'='*100}")
    conn.close()


def cmd_analyze(args):
    """Analyze collected spike data."""
    conn = init_db()

    print(f"\n{'='*100}")
    print(f"  SPIKE ANALYSIS")
    print(f"{'='*100}")

    total_markets = conn.execute("SELECT COUNT(*) FROM markets").fetchone()[0]
    total_spikes = conn.execute("SELECT COUNT(*) FROM spikes").fetchone()[0]
    total_prices = conn.execute("SELECT COUNT(*) FROM price_history").fetchone()[0]

    print(f"  Total markets:   {total_markets}")
    print(f"  Total spikes:    {total_spikes}")
    print(f"  Price datapoints:{total_prices:,}")

    if total_spikes == 0:
        print(f"\n  No spikes collected yet. Run: python -m engines.spike_scanner collect")
        conn.close()
        return

    # Spike stats by event type
    print(f"\n  Spikes by Event Type:")
    print(f"  {'Type':<15s} {'Count':>6s} {'AvgMove':>8s} {'AvgDur':>8s} {'AvgRevert':>10s} "
          f"{'MedMove':>8s}")
    print(f"  {'─'*65}")

    types = conn.execute("""
        SELECT event_type, COUNT(*), AVG(ABS(spike_pct)), AVG(spike_duration_mins),
               AVG(reversion_pct)
        FROM spikes GROUP BY event_type ORDER BY COUNT(*) DESC
    """).fetchall()

    for t in types:
        median = conn.execute("""
            SELECT ABS(spike_pct) FROM spikes WHERE event_type=?
            ORDER BY ABS(spike_pct) LIMIT 1 OFFSET ?
        """, (t[0], t[1] // 2)).fetchone()
        med_str = f"{median[0]:>7.1f}%" if median else "—"
        print(f"  {t[0]:<15s} {t[1]:>6d} {t[2]:>+7.1f}% {t[3]:>7.0f}m "
              f"{t[4]:>+9.1f}% {med_str}")

    # Spike direction distribution
    print(f"\n  Direction Distribution:")
    for direction in ["UP", "DOWN"]:
        count = conn.execute(
            "SELECT COUNT(*) FROM spikes WHERE direction=?", (direction,)).fetchone()[0]
        avg_move = conn.execute(
            "SELECT AVG(ABS(spike_pct)) FROM spikes WHERE direction=?", (direction,)).fetchone()[0]
        avg_revert = conn.execute(
            "SELECT AVG(ABS(reversion_pct)) FROM spikes WHERE direction=?", (direction,)).fetchone()[0]
        print(f"    {direction}: {count} spikes, avg move {avg_move or 0:.1f}%, "
              f"avg reversion {avg_revert or 0:.1f}%")

    # Reversion analysis — key for the predictor
    print(f"\n  Reversion Analysis (do spikes mean-revert?):")
    for bucket in [(5, 10), (10, 20), (20, 50), (50, 100)]:
        lo, hi = bucket
        rows = conn.execute("""
            SELECT COUNT(*), AVG(reversion_pct), AVG(reversion_mins)
            FROM spikes WHERE ABS(spike_pct) >= ? AND ABS(spike_pct) < ?
        """, (lo, hi)).fetchone()
        if rows[0] > 0:
            print(f"    {lo}-{hi}% spikes: {rows[0]:>4d} events, "
                  f"avg reversion {rows[1] or 0:+.1f}% in {rows[2] or 0:.0f}m")

    conn.close()
    print(f"\n{'='*100}")


def cmd_taxonomy(args):
    """Show event taxonomy breakdown."""
    conn = init_db()

    print(f"\n{'='*80}")
    print(f"  EVENT TAXONOMY")
    print(f"{'='*80}")

    types = conn.execute("""
        SELECT event_type, COUNT(*), SUM(num_spikes),
               AVG(volume), MAX(volume)
        FROM markets GROUP BY event_type ORDER BY SUM(num_spikes) DESC
    """).fetchall()

    print(f"\n  {'Type':<15s} {'Markets':>8s} {'Spikes':>8s} {'AvgVol':>12s} {'MaxVol':>14s}")
    print(f"  {'─'*65}")

    for t in types:
        print(f"  {t[0]:<15s} {t[1]:>8d} {t[2] or 0:>8.0f} "
              f"${t[3] or 0:>10,.0f} ${t[4] or 0:>12,.0f}")

    # Show example spikes per type
    print(f"\n  Example Spikes by Type:")
    for t in types[:8]:
        etype = t[0]
        examples = conn.execute("""
            SELECT question, spike_pct, spike_duration_mins, direction
            FROM spikes WHERE event_type=?
            ORDER BY ABS(spike_pct) DESC LIMIT 3
        """, (etype,)).fetchall()

        if examples:
            print(f"\n  [{etype}]")
            for ex in examples:
                print(f"    {ex[3]} {ex[1]:+.1f}% in {ex[2]:.0f}m — {ex[0][:60]}")

    conn.close()
    print(f"\n{'='*80}")


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Spike Scanner")
    subparsers = parser.add_subparsers(dest="command")

    p_collect = subparsers.add_parser("collect", help="Collect spike data")
    p_collect.add_argument("--days", type=int, default=14)
    p_collect.add_argument("--min-volume", type=float, default=10000)
    p_collect.add_argument("--max-markets", type=int, default=200)
    p_collect.add_argument("--fidelity", type=int, default=60,
                           help="Candle interval in seconds (default: 60)")

    p_analyze = subparsers.add_parser("analyze", help="Analyze spikes")
    p_taxonomy = subparsers.add_parser("taxonomy", help="Event taxonomy")

    args = parser.parse_args()

    if args.command == "collect":
        cmd_collect(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "taxonomy":
        cmd_taxonomy(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
