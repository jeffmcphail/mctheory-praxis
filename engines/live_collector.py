"""
engines/live_collector.py -- Live Forward Price Collector

Background process that continuously collects price snapshots for active
high-volume Polymarket markets. Builds the training dataset for the
event-driven spike predictor, especially for geopolitical/political/economic
markets that don't have CLOB historical data.

Runs as a background task (Windows Task Scheduler or persistent terminal).
Stores 1-minute price snapshots in SQLite.

Architecture:
    1. Every REFRESH_INTERVAL: discover active markets from Gamma API
    2. Every SAMPLE_INTERVAL: sample CLOB prices for tracked markets
    3. Rotate markets based on volume/activity changes
    4. Detect spikes in real-time and flag them

Usage:
    python -m engines.live_collector start                    # Start collecting
    python -m engines.live_collector start --top 100          # Track top 100 by volume
    python -m engines.live_collector start --interval 60      # Sample every 60s
    python -m engines.live_collector stats                    # Show collection stats
    python -m engines.live_collector export                   # Export to spike_scanner DB
"""
import argparse
import json
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
DB_PATH = Path("data/live_collector.db")

# Default parameters
DEFAULT_TOP_N = 50          # Track top N markets by volume
DEFAULT_SAMPLE_INTERVAL = 60  # Sample every 60 seconds
DEFAULT_REFRESH_INTERVAL = 900  # Refresh market list every 15 minutes
MIN_VOLUME = 50000          # Minimum volume to track


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")  # Better for concurrent reads
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tracked_markets (
            slug TEXT PRIMARY KEY,
            question TEXT,
            event_type TEXT,
            volume REAL,
            liquidity REAL,
            yes_token TEXT,
            no_token TEXT,
            end_date TEXT,
            neg_risk INTEGER,
            first_tracked TEXT,
            last_updated TEXT,
            active INTEGER DEFAULT 1
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS price_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slug TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            yes_mid REAL,
            yes_bid REAL,
            yes_ask REAL,
            spread REAL,
            UNIQUE(slug, timestamp)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS collection_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            markets_tracked INTEGER,
            samples_taken INTEGER,
            errors INTEGER,
            duration_ms REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS spike_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slug TEXT,
            question TEXT,
            event_type TEXT,
            detected_at TEXT,
            price_before REAL,
            price_now REAL,
            move_pct REAL,
            window_mins INTEGER
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ps_slug_ts ON price_snapshots(slug, timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ps_ts ON price_snapshots(timestamp)")
    conn.commit()
    return conn


def get_clob_midpoint(token_id):
    """Get midpoint price from CLOB. Returns float or 0."""
    try:
        r = requests.get(f"{CLOB_API}/midpoint",
                         params={"token_id": token_id}, timeout=5)
        if r.status_code == 200:
            data = r.json()
            return float(data.get("mid", 0)) if isinstance(data, dict) else float(data)
    except Exception:
        pass
    return 0


def get_clob_prices(token_id):
    """Get bid/ask/mid from CLOB. Returns dict."""
    result = {"mid": 0, "bid": 0, "ask": 0, "spread": 0}
    try:
        # Midpoint
        r = requests.get(f"{CLOB_API}/midpoint",
                         params={"token_id": token_id}, timeout=5)
        if r.status_code == 200:
            data = r.json()
            result["mid"] = float(data.get("mid", 0)) if isinstance(data, dict) else float(data)

        # Buy price (what you'd pay)
        r = requests.get(f"{CLOB_API}/price",
                         params={"token_id": token_id, "side": "BUY"}, timeout=5)
        if r.status_code == 200:
            result["bid"] = float(r.json().get("price", 0))

        # Sell price
        r = requests.get(f"{CLOB_API}/price",
                         params={"token_id": token_id, "side": "SELL"}, timeout=5)
        if r.status_code == 200:
            result["ask"] = float(r.json().get("price", 0))

        if result["ask"] > 0 and result["bid"] > 0:
            result["spread"] = result["ask"] - result["bid"]

    except Exception:
        pass
    return result


def refresh_market_list(conn, top_n=DEFAULT_TOP_N, verbose=True):
    """Refresh the list of tracked markets from Gamma API."""
    if verbose:
        print(f"    Refreshing market list (top {top_n} by volume)...")

    all_markets = []
    offset = 0
    limit = 100

    while len(all_markets) < top_n * 3:
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
            if len(batch) < limit:
                break
        except Exception:
            break

    # Filter and sort by volume
    valid = []
    for m in all_markets:
        vol = float(m.get("volume", 0) or 0)
        if vol < MIN_VOLUME:
            continue
        token_ids = json.loads(m.get("clobTokenIds", "[]"))
        if len(token_ids) < 1:
            continue
        valid.append(m)

    valid.sort(key=lambda m: float(m.get("volume", 0) or 0), reverse=True)
    top_markets = valid[:top_n]

    # Mark all current as inactive, then reactivate the ones we're tracking
    conn.execute("UPDATE tracked_markets SET active=0")

    now = datetime.now(timezone.utc).isoformat()
    new_count = 0

    for m in top_markets:
        slug = m.get("slug", "")
        token_ids = json.loads(m.get("clobTokenIds", "[]"))
        yes_token = token_ids[0] if token_ids else ""
        no_token = token_ids[1] if len(token_ids) > 1 else ""

        # Try to get event type from taxonomy (if classifier has run)
        event_type = "unknown"
        try:
            spike_db = sqlite3.connect("data/spike_scanner.db")
            row = spike_db.execute(
                "SELECT COALESCE(corrected_to, classified_as) FROM taxonomy WHERE slug=?",
                (slug,)).fetchone()
            if row:
                event_type = row[0]
            spike_db.close()
        except Exception:
            pass

        existing = conn.execute(
            "SELECT slug FROM tracked_markets WHERE slug=?", (slug,)).fetchone()

        if existing:
            conn.execute("""
                UPDATE tracked_markets SET
                    volume=?, liquidity=?, active=1, last_updated=?,
                    yes_token=?, no_token=?, event_type=?
                WHERE slug=?
            """, (
                float(m.get("volume", 0) or 0),
                float(m.get("liquidityClob", 0) or 0),
                now, yes_token, no_token, event_type, slug
            ))
        else:
            new_count += 1
            conn.execute("""
                INSERT INTO tracked_markets
                (slug, question, event_type, volume, liquidity,
                 yes_token, no_token, end_date, neg_risk,
                 first_tracked, last_updated, active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """, (
                slug, m.get("question", "")[:200], event_type,
                float(m.get("volume", 0) or 0),
                float(m.get("liquidityClob", 0) or 0),
                yes_token, no_token,
                m.get("endDate", ""),
                1 if m.get("negRisk") else 0,
                now, now
            ))

    conn.commit()

    active = conn.execute(
        "SELECT COUNT(*) FROM tracked_markets WHERE active=1").fetchone()[0]
    if verbose:
        print(f"    Tracking {active} markets ({new_count} new)")

    return active


def check_for_spikes(conn, slug, question, event_type, window_mins=60, threshold_pct=8.0):
    """Check if a market has spiked recently. Returns spike dict or None."""
    now = int(time.time())
    window_start = now - (window_mins * 60)

    prices = conn.execute("""
        SELECT timestamp, yes_mid FROM price_snapshots
        WHERE slug=? AND timestamp >= ? AND yes_mid > 0.05 AND yes_mid < 0.95
        ORDER BY timestamp ASC
    """, (slug, window_start)).fetchall()

    if len(prices) < 3:
        return None

    # Check if latest price vs earliest price in window exceeds threshold
    first_price = prices[0][1]
    last_price = prices[-1][1]

    if first_price <= 0:
        return None

    move_pct = (last_price - first_price) / first_price * 100

    if abs(move_pct) >= threshold_pct:
        return {
            "slug": slug,
            "question": question,
            "event_type": event_type,
            "price_before": first_price,
            "price_now": last_price,
            "move_pct": move_pct,
            "window_mins": window_mins,
        }

    return None


def sample_all_markets(conn, verbose=False):
    """Take a price snapshot for all active tracked markets."""
    markets = conn.execute("""
        SELECT slug, question, event_type, yes_token
        FROM tracked_markets WHERE active=1
    """).fetchall()

    now = int(time.time())
    samples = 0
    errors = 0
    spikes = []

    for slug, question, event_type, yes_token in markets:
        if not yes_token:
            continue

        try:
            mid = get_clob_midpoint(yes_token)
            if mid > 0:
                conn.execute("""
                    INSERT OR IGNORE INTO price_snapshots
                    (slug, timestamp, yes_mid)
                    VALUES (?, ?, ?)
                """, (slug, now, mid))
                samples += 1

                # Check for spikes
                spike = check_for_spikes(conn, slug, question, event_type)
                if spike:
                    # Check if we already alerted this spike (within last hour)
                    recent = conn.execute("""
                        SELECT id FROM spike_alerts
                        WHERE slug=? AND detected_at > ?
                    """, (slug, (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat())).fetchone()

                    if not recent:
                        spikes.append(spike)
                        conn.execute("""
                            INSERT INTO spike_alerts
                            (slug, question, event_type, detected_at,
                             price_before, price_now, move_pct, window_mins)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            slug, question[:100], event_type,
                            datetime.now(timezone.utc).isoformat(),
                            spike["price_before"], spike["price_now"],
                            spike["move_pct"], spike["window_mins"],
                        ))
        except Exception:
            errors += 1

    conn.commit()
    return samples, errors, spikes


# =======================================================
# COMMANDS
# =======================================================

def cmd_start(args):
    """Start the live collector."""
    top_n = getattr(args, "top", DEFAULT_TOP_N)
    sample_interval = getattr(args, "interval", DEFAULT_SAMPLE_INTERVAL)
    refresh_interval = getattr(args, "refresh", DEFAULT_REFRESH_INTERVAL)

    conn = init_db()

    print(f"\n{'='*90}")
    print(f"  LIVE FORWARD COLLECTOR")
    print(f"  Tracking top {top_n} markets | Sample every {sample_interval}s | "
          f"Refresh list every {refresh_interval}s")
    print(f"  Press Ctrl+C to stop.")
    print(f"{'='*90}")

    # Initial market list
    active = refresh_market_list(conn, top_n, verbose=True)
    last_refresh = time.time()

    # Show type distribution
    types = conn.execute("""
        SELECT event_type, COUNT(*) FROM tracked_markets
        WHERE active=1 GROUP BY event_type ORDER BY COUNT(*) DESC
    """).fetchall()
    print(f"\n  Event type distribution:")
    for t, c in types:
        print(f"    {t:<15s} {c:>4d}")

    print(f"\n  {'Time':<10s} {'Samples':>8s} {'Errors':>7s} {'Total':>8s} {'Spikes':>7s}")
    print(f"  {'-'*50}")

    cycle = 0
    total_samples = 0
    total_errors = 0
    total_spikes = 0

    while True:
        try:
            cycle += 1
            cycle_start = time.time()

            # Refresh market list periodically
            if time.time() - last_refresh >= refresh_interval:
                active = refresh_market_list(conn, top_n, verbose=True)
                last_refresh = time.time()

            # Sample all markets
            samples, errors, spikes = sample_all_markets(conn)
            total_samples += samples
            total_errors += errors
            total_spikes += len(spikes)

            cycle_ms = (time.time() - cycle_start) * 1000

            # Log
            conn.execute("""
                INSERT INTO collection_log
                (timestamp, markets_tracked, samples_taken, errors, duration_ms)
                VALUES (?, ?, ?, ?, ?)
            """, (
                datetime.now(timezone.utc).isoformat(),
                active, samples, errors, cycle_ms
            ))
            conn.commit()

            # Display
            now_str = datetime.now().strftime("%H:%M:%S")
            spike_str = f" !! {len(spikes)}" if spikes else ""
            print(f"  {now_str:<10s} {samples:>8d} {errors:>7d} "
                  f"{total_samples:>8d} {total_spikes:>7d}{spike_str}")

            # Show spike alerts
            for spike in spikes:
                direction = "UP" if spike["move_pct"] > 0 else "DN"
                print(f"    {direction} SPIKE: {spike['question'][:55]} "
                      f"| {spike['move_pct']:+.1f}% "
                      f"| {spike['price_before']:.3f}->{spike['price_now']:.3f} "
                      f"| [{spike['event_type']}]")

            # Wait for next cycle
            elapsed = time.time() - cycle_start
            sleep_time = max(0, sample_interval - elapsed)
            time.sleep(sleep_time)

        except KeyboardInterrupt:
            print(f"\n\n  Stopped after {cycle} cycles.")
            print(f"  Total samples: {total_samples:,}")
            print(f"  Total spikes detected: {total_spikes}")
            print(f"  Database: {DB_PATH}")
            break
        except Exception as e:
            print(f"  !! Cycle error: {e}")
            time.sleep(sample_interval)

    conn.close()


def cmd_stats(args):
    """Show collection statistics."""
    if not DB_PATH.exists():
        print(f"  No database. Run 'start' first.")
        return

    conn = sqlite3.connect(str(DB_PATH))

    total_snapshots = conn.execute("SELECT COUNT(*) FROM price_snapshots").fetchone()[0]
    total_markets = conn.execute("SELECT COUNT(*) FROM tracked_markets").fetchone()[0]
    active_markets = conn.execute(
        "SELECT COUNT(*) FROM tracked_markets WHERE active=1").fetchone()[0]
    total_spikes = conn.execute("SELECT COUNT(*) FROM spike_alerts").fetchone()[0]

    print(f"\n{'='*80}")
    print(f"  LIVE COLLECTOR STATS")
    print(f"{'='*80}")
    print(f"  Total markets tracked:  {total_markets}")
    print(f"  Currently active:       {active_markets}")
    print(f"  Price snapshots:        {total_snapshots:,}")
    print(f"  Spike alerts:           {total_spikes}")

    if total_snapshots > 0:
        first = conn.execute("SELECT MIN(timestamp) FROM price_snapshots").fetchone()[0]
        last = conn.execute("SELECT MAX(timestamp) FROM price_snapshots").fetchone()[0]
        if first and last:
            first_dt = datetime.fromtimestamp(first, tz=timezone.utc)
            last_dt = datetime.fromtimestamp(last, tz=timezone.utc)
            duration = last_dt - first_dt
            print(f"  Collection period:      {first_dt.strftime('%Y-%m-%d %H:%M')} -> "
                  f"{last_dt.strftime('%Y-%m-%d %H:%M')}")
            print(f"  Duration:               {duration}")

    # Type distribution
    types = conn.execute("""
        SELECT tm.event_type, COUNT(DISTINCT tm.slug), COUNT(*)
        FROM tracked_markets tm
        JOIN price_snapshots ps ON tm.slug = ps.slug
        WHERE tm.active=1
        GROUP BY tm.event_type
        ORDER BY COUNT(*) DESC
    """).fetchall()

    if types:
        print(f"\n  Snapshots by Event Type:")
        print(f"  {'Type':<15s} {'Markets':>8s} {'Snapshots':>10s}")
        print(f"  {'-'*35}")
        for t in types:
            print(f"  {t[0]:<15s} {t[1]:>8d} {t[2]:>10,d}")

    # Recent spike alerts
    recent_spikes = conn.execute("""
        SELECT detected_at, question, event_type, move_pct,
               price_before, price_now
        FROM spike_alerts ORDER BY detected_at DESC LIMIT 10
    """).fetchall()

    if recent_spikes:
        print(f"\n  Recent Spike Alerts:")
        print(f"  {'Time':<20s} {'Market':<40s} {'Type':<12s} {'Move':>7s}")
        print(f"  {'-'*85}")
        for s in recent_spikes:
            t = s[0][:19]
            print(f"  {t:<20s} {s[1][:39]:<40s} {s[2]:<12s} {s[3]:>+6.1f}%")

    conn.close()
    print(f"\n{'='*80}")


def cmd_export(args):
    """Export collected data to spike_scanner database for model training."""
    if not DB_PATH.exists():
        print(f"  No live collector database.")
        return

    spike_db_path = Path("data/spike_scanner.db")
    if not spike_db_path.exists():
        print(f"  No spike scanner database. Run spike_scanner collect first.")
        return

    live_conn = sqlite3.connect(str(DB_PATH))
    spike_conn = sqlite3.connect(str(spike_db_path))

    # Export price snapshots
    snapshots = live_conn.execute("""
        SELECT tm.slug, ps.timestamp, ps.yes_mid
        FROM price_snapshots ps
        JOIN tracked_markets tm ON ps.slug = tm.slug
    """).fetchall()

    exported = 0
    for slug, ts, price in snapshots:
        try:
            spike_conn.execute(
                "INSERT OR IGNORE INTO price_history (slug, timestamp, price) VALUES (?, ?, ?)",
                (slug, ts, price))
            exported += 1
        except Exception:
            pass

    spike_conn.commit()

    print(f"  Exported {exported:,} snapshots to spike_scanner.db")

    live_conn.close()
    spike_conn.close()


def main():
    parser = argparse.ArgumentParser(description="Live Forward Collector")
    subs = parser.add_subparsers(dest="command")

    p_start = subs.add_parser("start", help="Start collecting")
    p_start.add_argument("--top", type=int, default=DEFAULT_TOP_N,
                         help=f"Track top N markets (default: {DEFAULT_TOP_N})")
    p_start.add_argument("--interval", type=int, default=DEFAULT_SAMPLE_INTERVAL,
                         help=f"Sample interval in seconds (default: {DEFAULT_SAMPLE_INTERVAL})")
    p_start.add_argument("--refresh", type=int, default=DEFAULT_REFRESH_INTERVAL,
                         help=f"Market list refresh interval (default: {DEFAULT_REFRESH_INTERVAL})")

    subs.add_parser("stats", help="Show collection stats")
    subs.add_parser("export", help="Export to spike_scanner DB")

    args = parser.parse_args()

    if args.command == "start":
        cmd_start(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "export":
        cmd_export(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
