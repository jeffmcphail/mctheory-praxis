"""
engines/smart_money.py — Polymarket Smart Money Tracker

Monitors top leaderboard wallets for position changes. When a top trader
enters or exits a market, that's a signal (SPOILER) we can use before
placing our own trades.

Architecture:
    1. DISCOVERY: Pull top traders from leaderboard by PNL/category
    2. SNAPSHOT: Capture their current positions regularly
    3. DIFF: Compare snapshots to detect new entries/exits
    4. SIGNAL: Flag when multiple top traders converge on same market
    5. ALERT: Notify when high-conviction signals appear

Data sources (all public, no auth):
    - Polymarket Data API: leaderboard, positions, trades, activity
    - On-chain: CTF token balances (verification)

Usage:
    python -m engines.smart_money discover                     # Find top traders
    python -m engines.smart_money discover --category sports   # By category
    python -m engines.smart_money snapshot                     # Snapshot all tracked wallets
    python -m engines.smart_money diff                         # Show position changes
    python -m engines.smart_money signals                      # Show convergence signals
    python -m engines.smart_money monitor                      # Continuous monitoring
    python -m engines.smart_money profile 0xABC...             # Deep dive on a wallet
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

DATA_API = "https://data-api.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"
DB_PATH = Path("data/smart_money.db")

# Tracker config
DEFAULT_TOP_N = 25           # Track top N traders
MIN_POSITION_USD = 100       # Ignore positions under $100
SNAPSHOT_INTERVAL = 300      # Snapshot every 5 minutes
CONVERGENCE_THRESHOLD = 3    # N traders in same market = strong signal

CATEGORIES = ["OVERALL", "POLITICS", "SPORTS", "CRYPTO", "CULTURE"]


# ═══════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS tracked_wallets (
            address TEXT PRIMARY KEY,
            username TEXT,
            category TEXT,
            leaderboard_rank INTEGER,
            total_pnl REAL,
            volume REAL,
            markets_traded INTEGER,
            win_rate REAL,
            first_tracked TEXT,
            last_updated TEXT,
            active INTEGER DEFAULT 1
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS position_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            wallet TEXT NOT NULL,
            market_slug TEXT,
            market_title TEXT,
            outcome TEXT,
            size REAL,
            avg_price REAL,
            current_price REAL,
            value_usd REAL,
            pnl_usd REAL,
            UNIQUE(snapshot_id, wallet, market_slug, outcome)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS position_changes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detected_at TEXT NOT NULL,
            wallet TEXT NOT NULL,
            username TEXT,
            change_type TEXT,
            market_slug TEXT,
            market_title TEXT,
            outcome TEXT,
            old_size REAL,
            new_size REAL,
            size_delta REAL,
            price_at_change REAL,
            value_usd REAL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS convergence_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detected_at TEXT NOT NULL,
            market_slug TEXT,
            market_title TEXT,
            outcome TEXT,
            n_wallets INTEGER,
            wallets TEXT,
            avg_size REAL,
            total_value REAL,
            signal_strength REAL,
            current_price REAL
        )
    """)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_ps_snap ON position_snapshots(snapshot_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ps_wallet ON position_snapshots(wallet)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pc_market ON position_changes(market_slug)")

    conn.commit()
    return conn


# ═══════════════════════════════════════════════════════
# API HELPERS
# ═══════════════════════════════════════════════════════

def fetch_leaderboard(category="OVERALL", period="MONTH", limit=25, offset=0):
    """Fetch leaderboard from Polymarket Data API."""
    try:
        r = requests.get(f"{DATA_API}/v1/leaderboard", params={
            "limit": limit,
            "offset": offset,
            "timePeriod": period,
            "orderBy": "PNL",
            "category": category,
        }, timeout=15)

        if r.status_code == 200:
            return r.json()
        return []
    except Exception as e:
        print(f"    ❌ Leaderboard fetch failed: {e}")
        return []


def fetch_positions(wallet_address):
    """Fetch current positions for a wallet."""
    try:
        r = requests.get(f"{DATA_API}/positions", params={
            "user": wallet_address,
        }, timeout=15)

        if r.status_code == 200:
            return r.json()
        return []
    except Exception as e:
        print(f"    ❌ Position fetch failed for {wallet_address[:10]}...: {e}")
        return []


def fetch_trades(wallet_address, limit=50):
    """Fetch recent trades for a wallet."""
    try:
        r = requests.get(f"{DATA_API}/trades", params={
            "user": wallet_address,
            "limit": limit,
        }, timeout=15)

        if r.status_code == 200:
            return r.json()
        return []
    except Exception as e:
        return []


def fetch_activity(wallet_address):
    """Fetch activity feed for a wallet."""
    try:
        r = requests.get(f"{DATA_API}/activity", params={
            "user": wallet_address,
        }, timeout=15)

        if r.status_code == 200:
            return r.json()
        return []
    except Exception as e:
        return []


# ═══════════════════════════════════════════════════════
# COMMANDS
# ═══════════════════════════════════════════════════════

def cmd_discover(args):
    """Discover and register top traders from the leaderboard."""
    category = getattr(args, "category", "OVERALL").upper()
    top_n = getattr(args, "top", DEFAULT_TOP_N)
    period = getattr(args, "period", "MONTH").upper()

    conn = init_db()

    print(f"\n{'='*90}")
    print(f"  SMART MONEY DISCOVERY")
    print(f"  Category: {category} | Period: {period} | Top {top_n}")
    print(f"{'='*90}")

    if category == "ALL":
        categories = CATEGORIES
    else:
        categories = [category]

    all_traders = []

    for cat in categories:
        print(f"\n  Fetching {cat} leaderboard...")
        traders = fetch_leaderboard(category=cat, period=period,
                                     limit=top_n, offset=0)

        if not traders:
            print(f"    No results for {cat}")
            continue

        # Handle different response formats
        if isinstance(traders, dict):
            trader_list = traders.get("leaderboard", traders.get("data", []))
        elif isinstance(traders, list):
            trader_list = traders
        else:
            print(f"    Unexpected response format: {type(traders)}")
            continue

        print(f"    Found {len(trader_list)} traders")

        for rank, t in enumerate(trader_list, 1):
            # Polymarket uses various field names across API versions
            address = (t.get("proxyWallet", "") or
                       t.get("userAddress", "") or
                       t.get("address", "") or
                       t.get("wallet", "") or
                       t.get("proxy_wallet", "") or "")
            username = (t.get("userName", "") or
                        t.get("displayName", "") or
                        t.get("username", "") or
                        t.get("name", "") or
                        t.get("userSlug", "") or
                        address[:10] if address else "?")
            pnl = float(t.get("pnl", t.get("totalPnl", t.get("profit", 0))) or 0)
            volume = float(t.get("vol", t.get("volume", t.get("totalVolume", t.get("amount_traded", 0)))) or 0)
            markets = int(t.get("marketsTraded", t.get("totalMarkets",
                          t.get("markets_traded", t.get("numMarkets", 0)))) or 0)

            if not address:
                continue

            all_traders.append({
                "address": address,
                "username": username,
                "category": cat,
                "rank": rank,
                "pnl": pnl,
                "volume": volume,
                "markets": markets,
            })

            # Upsert into tracked_wallets
            now = datetime.now(timezone.utc).isoformat()
            conn.execute("""
                INSERT INTO tracked_wallets
                (address, username, category, leaderboard_rank, total_pnl,
                 volume, markets_traded, first_tracked, last_updated, active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(address) DO UPDATE SET
                    username=?, category=?, leaderboard_rank=?,
                    total_pnl=?, volume=?, markets_traded=?,
                    last_updated=?, active=1
            """, (
                address, username, cat, rank, pnl, volume, markets,
                now, now,
                username, cat, rank, pnl, volume, markets, now,
            ))

        time.sleep(0.5)  # Rate limit between categories

    conn.commit()

    # Display results
    total = conn.execute(
        "SELECT COUNT(*) FROM tracked_wallets WHERE active=1").fetchone()[0]

    print(f"\n{'─'*90}")
    print(f"  TRACKED WALLETS: {total}")
    print(f"{'─'*90}")

    print(f"\n  {'Rank':>4s} {'Username':<20s} {'Cat':<10s} "
          f"{'PnL':>12s} {'Volume':>12s} {'Markets':>8s} {'Address'}")
    print(f"  {'─'*100}")

    for t in sorted(all_traders, key=lambda x: -x["pnl"])[:30]:
        print(f"  {t['rank']:>4d} {t['username'][:19]:<20s} {t['category']:<10s} "
              f"${t['pnl']:>11,.0f} ${t['volume']:>11,.0f} "
              f"{t['markets']:>8d} {t['address'][:16]}...")

    conn.close()
    print(f"\n{'='*90}")


def cmd_snapshot(args):
    """Take a position snapshot for all tracked wallets."""
    conn = init_db()

    wallets = conn.execute(
        "SELECT address, username FROM tracked_wallets WHERE active=1"
    ).fetchall()

    if not wallets:
        print("  No tracked wallets. Run: python -m engines.smart_money discover")
        return

    snapshot_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    now = datetime.now(timezone.utc).isoformat()

    print(f"\n{'='*90}")
    print(f"  POSITION SNAPSHOT — {snapshot_id}")
    print(f"  Tracking {len(wallets)} wallets")
    print(f"{'='*90}")

    total_positions = 0
    last_progress = time.time()

    for i, (address, username) in enumerate(wallets):
        positions = fetch_positions(address)

        if not positions:
            time.sleep(0.3)
            continue

        for p in positions:
            title = p.get("title", p.get("market", {}).get("question", "?"))
            slug = ""
            if isinstance(p.get("market"), dict):
                slug = p["market"].get("slug", "")

            outcome = p.get("outcome", "?")
            size = float(p.get("size", 0) or 0)
            avg_price = float(p.get("avgPrice", p.get("averagePrice", 0)) or 0)
            cur_price = float(p.get("curPrice", p.get("currentPrice", 0)) or 0)

            value = size * cur_price if cur_price > 0 else size * avg_price
            pnl = (cur_price - avg_price) * size if avg_price > 0 else 0

            if value < MIN_POSITION_USD:
                continue

            conn.execute("""
                INSERT OR REPLACE INTO position_snapshots
                (snapshot_id, timestamp, wallet, market_slug, market_title,
                 outcome, size, avg_price, current_price, value_usd, pnl_usd)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot_id, now, address, slug, str(title)[:100],
                outcome, size, avg_price, cur_price, value, pnl,
            ))
            total_positions += 1

        time.sleep(0.3)  # Rate limit

        # Progress
        now_t = time.time()
        if now_t - last_progress >= 15 or i == len(wallets) - 1:
            print(f"    [{datetime.now().strftime('%H:%M:%S')}] "
                  f"{i+1}/{len(wallets)} wallets | "
                  f"{total_positions} positions captured")
            last_progress = now_t

    conn.commit()

    print(f"\n  Snapshot {snapshot_id}: {total_positions} positions from "
          f"{len(wallets)} wallets")

    # Show top holdings across all tracked wallets
    top_markets = conn.execute("""
        SELECT market_title, outcome, COUNT(DISTINCT wallet) as n_wallets,
               SUM(value_usd) as total_value, AVG(current_price) as avg_price
        FROM position_snapshots
        WHERE snapshot_id=?
        GROUP BY market_slug, outcome
        HAVING n_wallets >= 2
        ORDER BY n_wallets DESC, total_value DESC
        LIMIT 20
    """, (snapshot_id,)).fetchall()

    if top_markets:
        print(f"\n  Markets with multiple top traders:")
        print(f"  {'Market':<45s} {'Side':<5s} {'Wallets':>8s} "
              f"{'Total$':>10s} {'Price':>6s}")
        print(f"  {'─'*80}")
        for m in top_markets:
            flag = "🚨" if m[2] >= CONVERGENCE_THRESHOLD else "  "
            print(f"  {flag}{m[0][:43]:<45s} {m[1]:<5s} {m[2]:>8d} "
                  f"${m[3]:>9,.0f} {m[4]:>5.0%}")

    conn.close()
    print(f"\n{'='*90}")


def cmd_diff(args):
    """Compare the two most recent snapshots to find position changes."""
    conn = init_db()

    # Get two most recent snapshot IDs
    snapshots = conn.execute("""
        SELECT DISTINCT snapshot_id FROM position_snapshots
        ORDER BY snapshot_id DESC LIMIT 2
    """).fetchall()

    if len(snapshots) < 2:
        print("  Need at least 2 snapshots to diff. Run snapshot again in a few minutes.")
        conn.close()
        return

    new_snap = snapshots[0][0]
    old_snap = snapshots[1][0]

    print(f"\n{'='*90}")
    print(f"  POSITION DIFF")
    print(f"  Old: {old_snap} → New: {new_snap}")
    print(f"{'='*90}")

    # Build position maps
    def get_positions(snap_id):
        rows = conn.execute("""
            SELECT wallet, market_slug, market_title, outcome, size,
                   current_price, value_usd
            FROM position_snapshots WHERE snapshot_id=?
        """, (snap_id,)).fetchall()
        positions = {}
        for r in rows:
            key = (r[0], r[1], r[3])  # (wallet, slug, outcome)
            positions[key] = {
                "wallet": r[0],
                "slug": r[1],
                "title": r[2],
                "outcome": r[3],
                "size": r[4],
                "price": r[5],
                "value": r[6],
            }
        return positions

    old_pos = get_positions(old_snap)
    new_pos = get_positions(new_snap)

    changes = []
    now = datetime.now(timezone.utc).isoformat()

    # New positions (in new but not in old)
    for key, pos in new_pos.items():
        if key not in old_pos:
            wallet_name = conn.execute(
                "SELECT username FROM tracked_wallets WHERE address=?",
                (pos["wallet"],)).fetchone()
            username = wallet_name[0] if wallet_name else pos["wallet"][:12]

            changes.append({
                "type": "NEW",
                "wallet": pos["wallet"],
                "username": username,
                "title": pos["title"],
                "outcome": pos["outcome"],
                "old_size": 0,
                "new_size": pos["size"],
                "delta": pos["size"],
                "price": pos["price"],
                "value": pos["value"],
            })

    # Closed positions (in old but not in new)
    for key, pos in old_pos.items():
        if key not in new_pos:
            wallet_name = conn.execute(
                "SELECT username FROM tracked_wallets WHERE address=?",
                (pos["wallet"],)).fetchone()
            username = wallet_name[0] if wallet_name else pos["wallet"][:12]

            changes.append({
                "type": "CLOSED",
                "wallet": pos["wallet"],
                "username": username,
                "title": pos["title"],
                "outcome": pos["outcome"],
                "old_size": pos["size"],
                "new_size": 0,
                "delta": -pos["size"],
                "price": pos["price"],
                "value": pos["value"],
            })

    # Changed positions (size changed)
    for key in set(old_pos.keys()) & set(new_pos.keys()):
        old_size = old_pos[key]["size"]
        new_size = new_pos[key]["size"]
        delta = new_size - old_size

        if abs(delta) / max(old_size, 1) > 0.05:  # >5% change
            wallet_name = conn.execute(
                "SELECT username FROM tracked_wallets WHERE address=?",
                (new_pos[key]["wallet"],)).fetchone()
            username = wallet_name[0] if wallet_name else new_pos[key]["wallet"][:12]

            change_type = "INCREASED" if delta > 0 else "DECREASED"
            changes.append({
                "type": change_type,
                "wallet": new_pos[key]["wallet"],
                "username": username,
                "title": new_pos[key]["title"],
                "outcome": new_pos[key]["outcome"],
                "old_size": old_size,
                "new_size": new_size,
                "delta": delta,
                "price": new_pos[key]["price"],
                "value": new_pos[key]["value"],
            })

    # Store changes
    for c in changes:
        conn.execute("""
            INSERT INTO position_changes
            (detected_at, wallet, username, change_type, market_slug,
             market_title, outcome, old_size, new_size, size_delta,
             price_at_change, value_usd)
            VALUES (?, ?, ?, ?, '', ?, ?, ?, ?, ?, ?, ?)
        """, (
            now, c["wallet"], c["username"], c["type"],
            c["title"][:100], c["outcome"],
            c["old_size"], c["new_size"], c["delta"],
            c["price"], c["value"],
        ))

    conn.commit()

    # Display
    if not changes:
        print(f"\n  No position changes detected.")
    else:
        print(f"\n  {len(changes)} position changes detected:")
        print(f"\n  {'Type':<10s} {'Trader':<15s} {'Market':<40s} "
              f"{'Side':<5s} {'Delta':>10s} {'Price':>6s} {'Value':>8s}")
        print(f"  {'─'*100}")

        for c in sorted(changes, key=lambda x: -abs(x["value"])):
            icon = {"NEW": "🟢", "CLOSED": "🔴",
                     "INCREASED": "📈", "DECREASED": "📉"}.get(c["type"], "  ")
            print(f"  {icon}{c['type']:<9s} {c['username'][:14]:<15s} "
                  f"{c['title'][:39]:<40s} {c['outcome']:<5s} "
                  f"{c['delta']:>+10.1f} {c['price']:>5.0%} "
                  f"${c['value']:>7,.0f}")

    # Check for convergence signals
    new_entries = [c for c in changes if c["type"] in ("NEW", "INCREASED")]
    market_groups = {}
    for c in new_entries:
        key = (c["title"], c["outcome"])
        if key not in market_groups:
            market_groups[key] = []
        market_groups[key].append(c)

    convergence = {k: v for k, v in market_groups.items()
                   if len(v) >= 2}

    if convergence:
        print(f"\n  🚨 CONVERGENCE SIGNALS:")
        for (title, outcome), traders in convergence.items():
            total_val = sum(t["value"] for t in traders)
            n = len(traders)
            names = ", ".join(t["username"][:10] for t in traders)
            strength = "STRONG" if n >= CONVERGENCE_THRESHOLD else "MODERATE"

            print(f"    [{strength}] {title[:50]} ({outcome})")
            print(f"      {n} traders: {names}")
            print(f"      Total value: ${total_val:,.0f}")

            # Store signal
            conn.execute("""
                INSERT INTO convergence_signals
                (detected_at, market_title, outcome, n_wallets,
                 wallets, total_value, signal_strength, current_price)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now, title[:100], outcome, n, names,
                total_val, n / CONVERGENCE_THRESHOLD,
                traders[0]["price"],
            ))

        conn.commit()

    conn.close()
    print(f"\n{'='*90}")


def cmd_signals(args):
    """Show historical convergence signals."""
    if not DB_PATH.exists():
        print("  No data. Run discover + snapshot + diff first.")
        return

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    signals = conn.execute("""
        SELECT * FROM convergence_signals
        ORDER BY detected_at DESC LIMIT 20
    """).fetchall()

    print(f"\n{'='*70}")
    print(f"  SMART MONEY CONVERGENCE SIGNALS")
    print(f"{'='*70}")

    if not signals:
        print(f"\n  No convergence signals yet.")
        print(f"  Run snapshot → wait → snapshot → diff to generate signals.")
    else:
        for s in signals:
            strength = "🚨 STRONG" if s["signal_strength"] >= 1.0 else "⚠️ MODERATE"
            print(f"\n  {strength} [{s['detected_at'][:19]}]")
            print(f"    Market: {s['market_title']}")
            print(f"    Side: {s['outcome']} @ {s['current_price']:.0%}")
            print(f"    Traders ({s['n_wallets']}): {s['wallets']}")
            print(f"    Total value: ${s['total_value']:,.0f}")

    conn.close()
    print(f"\n{'='*70}")


def cmd_monitor(args):
    """Continuous monitoring loop."""
    interval = getattr(args, "interval", SNAPSHOT_INTERVAL)

    print(f"\n{'='*90}")
    print(f"  SMART MONEY MONITOR — Continuous")
    print(f"  Interval: {interval}s | Press Ctrl+C to stop")
    print(f"{'='*90}")

    cycle = 0

    while True:
        try:
            cycle += 1
            now_str = datetime.now().strftime("%H:%M:%S")

            # Take snapshot
            print(f"\n  {now_str} — Cycle {cycle}: Taking snapshot...")

            # Reuse snapshot logic inline (simplified)
            conn = init_db()
            wallets = conn.execute(
                "SELECT address, username FROM tracked_wallets WHERE active=1"
            ).fetchall()

            if not wallets:
                print("    No tracked wallets. Run discover first.")
                conn.close()
                time.sleep(interval)
                continue

            snapshot_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            now = datetime.now(timezone.utc).isoformat()
            total_pos = 0

            for address, username in wallets:
                positions = fetch_positions(address)
                for p in positions or []:
                    title = p.get("title", p.get("market", {}).get("question", "?"))
                    slug = ""
                    if isinstance(p.get("market"), dict):
                        slug = p["market"].get("slug", "")

                    outcome = p.get("outcome", "?")
                    size = float(p.get("size", 0) or 0)
                    avg_price = float(p.get("avgPrice", p.get("averagePrice", 0)) or 0)
                    cur_price = float(p.get("curPrice", p.get("currentPrice", 0)) or 0)
                    value = size * cur_price if cur_price > 0 else size * avg_price

                    if value < MIN_POSITION_USD:
                        continue

                    conn.execute("""
                        INSERT OR REPLACE INTO position_snapshots
                        (snapshot_id, timestamp, wallet, market_slug, market_title,
                         outcome, size, avg_price, current_price, value_usd, pnl_usd)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        snapshot_id, now, address, slug, str(title)[:100],
                        outcome, size, avg_price, cur_price, value,
                        (cur_price - avg_price) * size,
                    ))
                    total_pos += 1
                time.sleep(0.2)

            conn.commit()
            print(f"    Snapshot {snapshot_id}: {total_pos} positions")

            # Auto-diff if we have 2+ snapshots
            snap_count = conn.execute(
                "SELECT COUNT(DISTINCT snapshot_id) FROM position_snapshots"
            ).fetchone()[0]

            if snap_count >= 2 and cycle > 1:
                # Quick diff (reuse logic from cmd_diff)
                snaps = conn.execute("""
                    SELECT DISTINCT snapshot_id FROM position_snapshots
                    ORDER BY snapshot_id DESC LIMIT 2
                """).fetchall()

                new_s, old_s = snaps[0][0], snaps[1][0]

                # Count changes
                new_pos_keys = set()
                old_pos_keys = set()

                for row in conn.execute(
                    "SELECT wallet, market_slug, outcome FROM position_snapshots WHERE snapshot_id=?",
                    (new_s,)):
                    new_pos_keys.add((row[0], row[1], row[2]))
                for row in conn.execute(
                    "SELECT wallet, market_slug, outcome FROM position_snapshots WHERE snapshot_id=?",
                    (old_s,)):
                    old_pos_keys.add((row[0], row[1], row[2]))

                new_entries = new_pos_keys - old_pos_keys
                closed = old_pos_keys - new_pos_keys

                if new_entries or closed:
                    print(f"    📊 Changes: {len(new_entries)} new, {len(closed)} closed")
                else:
                    print(f"    No position changes")

            conn.close()

            # Cleanup old snapshots (keep last 50)
            if cycle % 10 == 0:
                conn = init_db()
                conn.execute("""
                    DELETE FROM position_snapshots
                    WHERE snapshot_id NOT IN (
                        SELECT DISTINCT snapshot_id FROM position_snapshots
                        ORDER BY snapshot_id DESC LIMIT 50
                    )
                """)
                conn.commit()
                conn.close()

            time.sleep(interval)

        except KeyboardInterrupt:
            print(f"\n  Stopped after {cycle} cycles.")
            break
        except Exception as e:
            print(f"    ⚠️ Error: {e}")
            time.sleep(interval)


def cmd_profile(args):
    """Deep dive on a specific wallet."""
    address = args.address

    print(f"\n{'='*90}")
    print(f"  WALLET PROFILE: {address}")
    print(f"{'='*90}")

    # Fetch positions
    print(f"\n  Current Positions:")
    positions = fetch_positions(address)

    if not positions:
        print(f"    No positions found (or API error)")
    else:
        # Sort by value
        enriched = []
        for p in positions:
            title = p.get("title", p.get("market", {}).get("question", "?"))
            outcome = p.get("outcome", "?")
            size = float(p.get("size", 0) or 0)
            avg_price = float(p.get("avgPrice", p.get("averagePrice", 0)) or 0)
            cur_price = float(p.get("curPrice", p.get("currentPrice", 0)) or 0)
            value = size * cur_price if cur_price > 0 else size * avg_price
            pnl = (cur_price - avg_price) * size if avg_price > 0 else 0

            enriched.append({
                "title": str(title)[:55],
                "outcome": outcome,
                "size": size,
                "entry": avg_price,
                "current": cur_price,
                "value": value,
                "pnl": pnl,
            })

        enriched.sort(key=lambda x: -x["value"])

        print(f"  {'Market':<55s} {'Side':<5s} {'Size':>8s} "
              f"{'Entry':>6s} {'Now':>6s} {'Value':>9s} {'P&L':>9s}")
        print(f"  {'─'*105}")

        total_value = 0
        total_pnl = 0

        for p in enriched[:25]:
            icon = "✅" if p["pnl"] > 0 else "❌" if p["pnl"] < 0 else "  "
            print(f"  {p['title']:<55s} {p['outcome']:<5s} "
                  f"{p['size']:>8.1f} {p['entry']:>5.0%} {p['current']:>5.0%} "
                  f"${p['value']:>8,.0f} ${p['pnl']:>+8,.0f} {icon}")
            total_value += p["value"]
            total_pnl += p["pnl"]

        if len(enriched) > 25:
            print(f"  ... and {len(enriched) - 25} more positions")

        print(f"\n  Total: {len(enriched)} positions | "
              f"Value: ${total_value:,.0f} | P&L: ${total_pnl:+,.0f}")

    # Recent trades
    print(f"\n  Recent Trades:")
    trades = fetch_trades(address, limit=15)

    if trades:
        for t in trades[:15]:
            side = t.get("side", "?")
            price = float(t.get("price", 0) or 0)
            size = float(t.get("size", 0) or 0)
            title = t.get("market", t.get("title", "?"))
            if isinstance(title, dict):
                title = title.get("question", "?")
            ts = t.get("timestamp", t.get("createdAt", ""))[:19]

            print(f"    {ts} {side:<5s} {size:>8.1f} @ {price:.3f} "
                  f"| {str(title)[:45]}")
    else:
        print(f"    No recent trades found")

    print(f"\n{'='*90}")


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Smart Money Tracker")
    subs = parser.add_subparsers(dest="command")

    p_disc = subs.add_parser("discover", help="Find top traders")
    p_disc.add_argument("--category", default="OVERALL",
                        help="OVERALL, POLITICS, SPORTS, CRYPTO, CULTURE, or ALL")
    p_disc.add_argument("--top", type=int, default=DEFAULT_TOP_N)
    p_disc.add_argument("--period", default="MONTH",
                        help="MONTH, WEEK, ALL")

    subs.add_parser("snapshot", help="Snapshot all tracked positions")
    subs.add_parser("diff", help="Compare recent snapshots")
    subs.add_parser("signals", help="Show convergence signals")

    p_mon = subs.add_parser("monitor", help="Continuous monitoring")
    p_mon.add_argument("--interval", type=int, default=SNAPSHOT_INTERVAL)

    p_prof = subs.add_parser("profile", help="Deep dive on a wallet")
    p_prof.add_argument("address", type=str)

    args = parser.parse_args()

    if args.command == "discover":
        cmd_discover(args)
    elif args.command == "snapshot":
        cmd_snapshot(args)
    elif args.command == "diff":
        cmd_diff(args)
    elif args.command == "signals":
        cmd_signals(args)
    elif args.command == "monitor":
        cmd_monitor(args)
    elif args.command == "profile":
        cmd_profile(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
