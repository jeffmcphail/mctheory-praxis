"""
sentiment_tracker.py — Price History Collector & Overreaction Detector

Collects hourly price snapshots across all Polymarket markets.
Detects sharp moves (potential overreactions) for mean-reversion trades.

Place in engines/ directory.

Usage:
    # Collect current prices (run hourly via scheduler)
    python -m engines.sentiment_tracker collect

    # Detect overreactions in collected data
    python -m engines.sentiment_tracker detect

    # Full scan: collect + detect + show signals
    python -m engines.sentiment_tracker scan

    # Show price history for a specific market
    python -m engines.sentiment_tracker history --grep "iran"

    # Database stats
    python -m engines.sentiment_tracker stats
"""
import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone, timedelta

import requests

DB_PATH = "data/market_prices.db"
GAMMA_API = "https://gamma-api.polymarket.com"

# Detection thresholds
MOVE_THRESHOLD_24H = 0.15    # 15% move in 24 hours
MOVE_THRESHOLD_12H = 0.12    # 12% move in 12 hours
MOVE_THRESHOLD_6H = 0.08     # 8% move in 6 hours
MIN_VOLUME = 5000            # Minimum volume to consider
MIN_PRICE = 0.05             # Skip near-zero markets
MAX_PRICE = 0.95             # Skip near-certain markets


def init_db():
    """Initialize SQLite database for price snapshots."""
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS price_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            epoch INTEGER NOT NULL,
            condition_id TEXT NOT NULL,
            question TEXT,
            slug TEXT,
            yes_price REAL,
            no_price REAL,
            volume REAL,
            liquidity REAL,
            end_date TEXT,
            tags TEXT
        )
    """)

    c.execute("""
        CREATE INDEX IF NOT EXISTS idx_snapshots_condition
        ON price_snapshots(condition_id, epoch)
    """)

    c.execute("""
        CREATE INDEX IF NOT EXISTS idx_snapshots_epoch
        ON price_snapshots(epoch)
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS detected_moves (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detected_at TEXT NOT NULL,
            condition_id TEXT NOT NULL,
            question TEXT,
            slug TEXT,
            current_price REAL,
            price_before REAL,
            move_pct REAL,
            timeframe_hours REAL,
            volume REAL,
            direction TEXT,
            signal TEXT,
            traded INTEGER DEFAULT 0
        )
    """)

    conn.commit()
    return conn


def fetch_all_markets():
    """Fetch all active markets from Gamma API."""
    all_markets = []
    seen = set()

    tags = [
        "politics", "sports", "crypto", "finance", "tech", "culture",
        "geopolitics", "economy", "elections", "entertainment", "science",
        "business", "weather", "iran", "trump",
    ]

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
                            yes_price = float(prices[0])
                            volume = float(m.get("volume", 0))

                            all_markets.append({
                                "condition_id": cid,
                                "question": m.get("question", m.get("groupItemTitle", "")),
                                "slug": event.get("slug", ""),
                                "yes_price": yes_price,
                                "no_price": 1.0 - yes_price,
                                "volume": volume,
                                "liquidity": float(m.get("liquidity", 0)),
                                "end_date": m.get("endDate", ""),
                                "tags": tag,
                            })
                        except (json.JSONDecodeError, ValueError, IndexError):
                            continue
                if len(batch) < 100:
                    break
                time.sleep(0.2)
            except Exception:
                break

    return all_markets


def cmd_collect(args):
    """Collect price snapshots for all active markets."""
    conn = init_db()
    c = conn.cursor()

    now = datetime.now(timezone.utc)
    epoch = int(now.timestamp())
    ts = now.isoformat()

    # Check last collection time
    c.execute("SELECT MAX(epoch) FROM price_snapshots")
    last = c.fetchone()[0]
    if last and epoch - last < 300:  # Less than 5 minutes ago
        print(f"  Last collection was {epoch - last}s ago. Skipping (min 5 min).")
        conn.close()
        return

    print(f"\n  Collecting market prices at {now.strftime('%Y-%m-%d %H:%M UTC')}...")
    markets = fetch_all_markets()
    print(f"  Fetched {len(markets)} markets")

    # Insert snapshots
    inserted = 0
    for m in markets:
        if m["volume"] < 100:  # Skip very low volume
            continue
        c.execute("""
            INSERT INTO price_snapshots
            (timestamp, epoch, condition_id, question, slug, yes_price, no_price,
             volume, liquidity, end_date, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ts, epoch, m["condition_id"], m["question"][:200], m["slug"],
            m["yes_price"], m["no_price"], m["volume"], m["liquidity"],
            m["end_date"], m["tags"],
        ))
        inserted += 1

    conn.commit()

    # Stats
    c.execute("SELECT COUNT(DISTINCT epoch) FROM price_snapshots")
    total_snapshots = c.fetchone()[0]
    c.execute("SELECT MIN(epoch), MAX(epoch) FROM price_snapshots")
    min_epoch, max_epoch = c.fetchone()
    hours_of_data = (max_epoch - min_epoch) / 3600 if min_epoch and max_epoch else 0

    print(f"  Inserted {inserted} market prices")
    print(f"  Total snapshots: {total_snapshots} collections over {hours_of_data:.1f} hours")

    conn.close()


def cmd_detect(args):
    """Detect sharp price moves (potential overreactions)."""
    conn = init_db()
    c = conn.cursor()

    now_epoch = int(datetime.now(timezone.utc).timestamp())
    now_ts = datetime.now(timezone.utc).isoformat()

    # Get latest prices
    c.execute("""
        SELECT condition_id, question, slug, yes_price, volume, end_date, tags
        FROM price_snapshots
        WHERE epoch = (SELECT MAX(epoch) FROM price_snapshots)
    """)
    latest = {row[0]: {
        "condition_id": row[0], "question": row[1], "slug": row[2],
        "yes_price": row[3], "volume": row[4], "end_date": row[5], "tags": row[6],
    } for row in c.fetchall()}

    if not latest:
        print("  No data collected yet. Run: python -m engines.sentiment_tracker collect")
        conn.close()
        return

    print(f"\n  Scanning {len(latest)} markets for overreactions...")

    # Check each timeframe
    timeframes = [
        (6, MOVE_THRESHOLD_6H),
        (12, MOVE_THRESHOLD_12H),
        (24, MOVE_THRESHOLD_24H),
    ]

    signals = []

    for hours, threshold in timeframes:
        target_epoch = now_epoch - int(hours * 3600)

        for cid, current in latest.items():
            cur_price = current["yes_price"]

            # Skip extreme prices and low volume
            if cur_price < MIN_PRICE or cur_price > MAX_PRICE:
                continue
            if current["volume"] < MIN_VOLUME:
                continue

            # Find price closest to target_epoch ago
            c.execute("""
                SELECT yes_price, epoch FROM price_snapshots
                WHERE condition_id = ? AND epoch <= ? AND epoch >= ?
                ORDER BY epoch DESC LIMIT 1
            """, (cid, target_epoch + 1800, target_epoch - 1800))
            row = c.fetchone()

            if not row:
                continue

            old_price = row[0]
            old_epoch = row[1]
            actual_hours = (now_epoch - old_epoch) / 3600

            if old_price < 0.01:
                continue

            move = (cur_price - old_price) / old_price

            if abs(move) >= threshold:
                direction = "SPIKE" if move > 0 else "CRASH"
                # Mean reversion signal: fade the move
                signal = "BUY NO" if move > 0 else "BUY YES"

                signals.append({
                    "condition_id": cid,
                    "question": current["question"],
                    "slug": current["slug"],
                    "current_price": cur_price,
                    "price_before": old_price,
                    "move_pct": move,
                    "timeframe_hours": actual_hours,
                    "volume": current["volume"],
                    "direction": direction,
                    "signal": signal,
                    "end_date": current["end_date"],
                })

    # Deduplicate (keep largest move per market)
    best_by_market = {}
    for s in signals:
        cid = s["condition_id"]
        if cid not in best_by_market or abs(s["move_pct"]) > abs(best_by_market[cid]["move_pct"]):
            best_by_market[cid] = s

    signals = sorted(best_by_market.values(), key=lambda s: abs(s["move_pct"]), reverse=True)

    # Display
    print(f"\n{'='*100}")
    print(f"  OVERREACTION SIGNALS — {len(signals)} detected")
    print(f"{'='*100}")

    if signals:
        print(f"\n  {'Question':<50s} {'Now':>5s} {'Was':>5s} {'Move':>7s} {'Hours':>6s} "
              f"{'Dir':<6s} {'Signal':<8s} {'Vol':>10s}")
        print(f"  {'-'*98}")

        for s in signals[:30]:
            q = s["question"][:49]
            print(f"  {q:<50s} {s['current_price']:>4.0%} {s['price_before']:>4.0%} "
                  f"{s['move_pct']:>+6.0%} {s['timeframe_hours']:>5.1f}h "
                  f"{s['direction']:<6s} {s['signal']:<8s} ${s['volume']:>9,.0f}")

        # Save to detected_moves table
        for s in signals:
            c.execute("""
                INSERT INTO detected_moves
                (detected_at, condition_id, question, slug, current_price,
                 price_before, move_pct, timeframe_hours, volume, direction, signal)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now_ts, s["condition_id"], s["question"][:200], s["slug"],
                s["current_price"], s["price_before"], s["move_pct"],
                s["timeframe_hours"], s["volume"], s["direction"], s["signal"],
            ))
        conn.commit()
    else:
        c.execute("SELECT COUNT(DISTINCT epoch) FROM price_snapshots")
        n = c.fetchone()[0]
        if n < 2:
            print(f"\n  Only {n} snapshot(s) collected. Need at least 2 collections")
            print(f"  (separated by hours) to detect moves.")
            print(f"  The collector runs hourly. Check back in a few hours.")
        else:
            print(f"\n  No overreactions detected above thresholds.")
            print(f"  Thresholds: ±{MOVE_THRESHOLD_6H:.0%}/6h, "
                  f"±{MOVE_THRESHOLD_12H:.0%}/12h, ±{MOVE_THRESHOLD_24H:.0%}/24h")

    conn.close()
    return signals


def cmd_history(args):
    """Show price history for a specific market."""
    conn = init_db()
    c = conn.cursor()

    if not args.grep:
        print("  Usage: --grep 'search term'")
        conn.close()
        return

    pattern = f"%{args.grep}%"
    c.execute("""
        SELECT DISTINCT condition_id, question
        FROM price_snapshots
        WHERE question LIKE ?
        ORDER BY volume DESC
        LIMIT 10
    """, (pattern,))

    markets = c.fetchall()
    if not markets:
        print(f"  No markets matching '{args.grep}'")
        conn.close()
        return

    for cid, question in markets:
        print(f"\n  {question[:80]}")
        print(f"  {'-'*80}")

        c.execute("""
            SELECT timestamp, yes_price, volume
            FROM price_snapshots
            WHERE condition_id = ?
            ORDER BY epoch ASC
        """, (cid,))
        rows = c.fetchall()

        print(f"  {'Time':<25s} {'YES Price':>10s} {'Change':>8s}")
        prev_price = None
        for ts, price, vol in rows:
            change = ""
            if prev_price is not None and prev_price > 0:
                diff = (price - prev_price) / prev_price
                change = f"{diff:>+7.1%}"
            print(f"  {ts[:19]:<25s} {price:>9.1%} {change}")
            prev_price = price

    conn.close()


def cmd_scan(args):
    """Full pipeline: collect + detect."""
    print(f"\n{'='*100}")
    print(f"SENTIMENT OVERREACTION SCANNER")
    print(f"{'='*100}")

    cmd_collect(args)
    print()
    signals = cmd_detect(args)

    if signals:
        print(f"\n  To view history: python -m engines.sentiment_tracker history --grep 'keyword'")
    print(f"\n  Next collection in ~1 hour (or run manually)")


def cmd_stats(args):
    """Show database statistics."""
    conn = init_db()
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM price_snapshots")
    total_rows = c.fetchone()[0]

    c.execute("SELECT COUNT(DISTINCT epoch) FROM price_snapshots")
    total_collections = c.fetchone()[0]

    c.execute("SELECT COUNT(DISTINCT condition_id) FROM price_snapshots")
    total_markets = c.fetchone()[0]

    c.execute("SELECT MIN(epoch), MAX(epoch) FROM price_snapshots")
    min_e, max_e = c.fetchone()
    hours = (max_e - min_e) / 3600 if min_e and max_e else 0

    c.execute("SELECT COUNT(*) FROM detected_moves")
    total_detections = c.fetchone()[0]

    print(f"\n  Database Statistics:")
    print(f"    Total rows:        {total_rows:,}")
    print(f"    Collections:       {total_collections}")
    print(f"    Unique markets:    {total_markets:,}")
    print(f"    Time span:         {hours:.1f} hours")
    print(f"    Detected moves:    {total_detections}")

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Sentiment Tracker")
    subs = parser.add_subparsers(dest="command", required=True)

    subs.add_parser("collect", help="Collect price snapshots")
    subs.add_parser("detect", help="Detect overreactions")
    subs.add_parser("scan", help="Collect + detect")
    subs.add_parser("stats", help="Database statistics")

    p_hist = subs.add_parser("history", help="Price history")
    p_hist.add_argument("--grep", type=str, required=True)

    args = parser.parse_args()

    dispatch = {
        "collect": cmd_collect,
        "detect": cmd_detect,
        "scan": cmd_scan,
        "stats": cmd_stats,
        "history": cmd_history,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
