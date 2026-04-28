"""
engines/smart_money_alerts.py -- Smart Money Diff, Alert & Trading System

Runs daily to:
  1. Snapshot current positions for all tracked wallets
  2. Diff against previous snapshot to find NEW entries
  3. Detect convergence: 3+ top traders entering the same market
  4. Score and rank convergence signals
  5. Optionally auto-execute trades on high-conviction signals

The edge hypothesis: top Polymarket traders have information or models
that let them identify mispriced markets. When multiple top traders
independently enter the same position, that's a strong directional signal.

Usage:
    python -m engines.smart_money_alerts diff                    # Show changes since last snapshot
    python -m engines.smart_money_alerts diff --hours 24         # Changes in last 24h
    python -m engines.smart_money_alerts convergence             # Current convergence signals
    python -m engines.smart_money_alerts convergence --min 3     # Min 3 wallets converging
    python -m engines.smart_money_alerts monitor                 # Continuous monitoring
    python -m engines.smart_money_alerts trade --signal "slug"   # Paper trade a signal
    python -m engines.smart_money_alerts trade --signal "slug" --live  # Real trade
    python -m engines.smart_money_alerts performance             # Track signal accuracy
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

DB_PATH = Path("data/smart_money.db")
ALERTS_DB_PATH = Path("data/smart_money_alerts.db")
DATA_API = "https://data-api.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

DEFAULT_MIN_CONVERGENCE = 3   # Min wallets for signal
DEFAULT_MIN_CAPITAL = 10000   # Min total $ for signal to matter
MAX_TRADE_SIZE = 25           # Max $ per auto-trade
SIGNAL_EXPIRY_HOURS = 48      # Signal expires after 48h


# ===================================================================
# DATABASE
# ===================================================================

def init_alerts_db():
    ALERTS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(ALERTS_DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS position_diffs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            diff_id TEXT NOT NULL,
            wallet TEXT NOT NULL,
            username TEXT,
            market_slug TEXT,
            question TEXT,
            outcome TEXT,
            action TEXT,
            size_change REAL,
            new_size REAL,
            price_at_diff REAL,
            dollar_value REAL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS convergence_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            signal_id TEXT NOT NULL,
            market_slug TEXT NOT NULL,
            question TEXT,
            outcome TEXT,
            n_wallets INTEGER,
            wallets TEXT,
            total_capital REAL,
            avg_entry_price REAL,
            market_price REAL,
            score REAL,
            status TEXT DEFAULT 'ACTIVE',
            trade_id TEXT,
            resolution TEXT,
            resolved_at TEXT,
            pnl REAL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS signal_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            signal_id TEXT NOT NULL,
            market_slug TEXT,
            outcome TEXT,
            side TEXT,
            size REAL,
            price REAL,
            cost REAL,
            mode TEXT DEFAULT 'PAPER',
            order_id TEXT,
            status TEXT DEFAULT 'PENDING',
            exit_price REAL,
            exit_time TEXT,
            pnl REAL
        )
    """)

    conn.commit()
    return conn


def get_smart_money_db():
    """Connect to the main smart_money.db."""
    return sqlite3.connect(str(DB_PATH))


# ===================================================================
# DIFF ENGINE
# ===================================================================

def get_latest_snapshots(conn, n=2):
    """Get the two most recent snapshot IDs."""
    rows = conn.execute("""
        SELECT DISTINCT snapshot_id FROM position_snapshots
        ORDER BY snapshot_id DESC LIMIT ?
    """, (n,)).fetchall()
    return [r[0] for r in rows]


def compute_diff(conn, snapshot_new, snapshot_old):
    """Compute position changes between two snapshots."""
    # Get positions from each snapshot
    new_positions = {}
    for row in conn.execute("""
        SELECT wallet, market_slug, outcome, size, current_price, market_title
        FROM position_snapshots
        WHERE snapshot_id=?
    """, (snapshot_new,)).fetchall():
        key = (row[0], row[1], row[2])
        new_positions[key] = {
            "wallet": row[0], "slug": row[1], "outcome": row[2],
            "size": row[3], "price": row[4] or 0, "question": row[5],
        }

    old_positions = {}
    for row in conn.execute("""
        SELECT wallet, market_slug, outcome, size, current_price, market_title
        FROM position_snapshots
        WHERE snapshot_id=?
    """, (snapshot_old,)).fetchall():
        key = (row[0], row[1], row[2])
        old_positions[key] = {
            "wallet": row[0], "slug": row[1], "outcome": row[2],
            "size": row[3], "price": row[4] or 0, "question": row[5],
        }

    diffs = []

    # New entries (in new but not old, or size increased)
    for key, new_pos in new_positions.items():
        old_pos = old_positions.get(key)
        if old_pos is None:
            # Brand new position
            diffs.append({
                "wallet": new_pos["wallet"],
                "slug": new_pos["slug"],
                "question": new_pos["question"],
                "outcome": new_pos["outcome"],
                "action": "ENTER",
                "size_change": new_pos["size"],
                "new_size": new_pos["size"],
                "price": new_pos["price"],
                "dollar_value": new_pos["size"] * new_pos["price"],
            })
        elif new_pos["size"] > old_pos["size"] * 1.05:
            # Size increased >5%
            change = new_pos["size"] - old_pos["size"]
            diffs.append({
                "wallet": new_pos["wallet"],
                "slug": new_pos["slug"],
                "question": new_pos["question"],
                "outcome": new_pos["outcome"],
                "action": "INCREASE",
                "size_change": change,
                "new_size": new_pos["size"],
                "price": new_pos["price"],
                "dollar_value": change * new_pos["price"],
            })

    # Exits (in old but not new, or size decreased)
    for key, old_pos in old_positions.items():
        new_pos = new_positions.get(key)
        if new_pos is None:
            diffs.append({
                "wallet": old_pos["wallet"],
                "slug": old_pos["slug"],
                "question": old_pos["question"],
                "outcome": old_pos["outcome"],
                "action": "EXIT",
                "size_change": -old_pos["size"],
                "new_size": 0,
                "price": old_pos["price"],
                "dollar_value": old_pos["size"] * old_pos["price"],
            })
        elif new_pos["size"] < old_pos["size"] * 0.95:
            change = new_pos["size"] - old_pos["size"]
            diffs.append({
                "wallet": old_pos["wallet"],
                "slug": old_pos["slug"],
                "question": old_pos["question"],
                "outcome": old_pos["outcome"],
                "action": "DECREASE",
                "size_change": change,
                "new_size": new_pos["size"],
                "price": new_pos["price"],
                "dollar_value": abs(change) * new_pos["price"],
            })

    return diffs


def detect_convergence(diffs, min_wallets=DEFAULT_MIN_CONVERGENCE):
    """Find markets where multiple wallets are entering the same side."""
    # Group entries by (slug, outcome)
    entries = {}
    for d in diffs:
        if d["action"] not in ("ENTER", "INCREASE"):
            continue
        key = (d["slug"], d["outcome"])
        if key not in entries:
            entries[key] = {
                "slug": d["slug"],
                "question": d["question"],
                "outcome": d["outcome"],
                "wallets": [],
                "total_capital": 0,
                "prices": [],
            }
        entries[key]["wallets"].append(d["wallet"])
        entries[key]["total_capital"] += d["dollar_value"]
        entries[key]["prices"].append(d["price"])

    # Filter for convergence
    signals = []
    for key, entry in entries.items():
        if len(entry["wallets"]) >= min_wallets:
            avg_price = sum(entry["prices"]) / len(entry["prices"])
            # Score: more wallets + more capital = stronger signal
            score = len(entry["wallets"]) * 10 + min(entry["total_capital"] / 10000, 50)

            signals.append({
                "slug": entry["slug"],
                "question": entry["question"],
                "outcome": entry["outcome"],
                "n_wallets": len(entry["wallets"]),
                "wallets": entry["wallets"],
                "total_capital": entry["total_capital"],
                "avg_entry_price": avg_price,
                "score": score,
            })

    signals.sort(key=lambda x: -x["score"])
    return signals


# ===================================================================
# COMMANDS
# ===================================================================

def cmd_diff(args):
    """Show position changes since last snapshot."""
    hours = getattr(args, "hours", None)

    sm_conn = get_smart_money_db()
    alerts_conn = init_alerts_db()

    snapshots = get_latest_snapshots(sm_conn)

    if len(snapshots) < 2:
        print(f"\n  Need at least 2 snapshots. Run:")
        print(f"  python -m engines.smart_money snapshot")
        print(f"  (wait, then run again)")
        return

    snap_new, snap_old = snapshots[0], snapshots[1]

    print(f"\n{'='*90}")
    print(f"  SMART MONEY POSITION DIFF")
    print(f"  Comparing: {snap_new} vs {snap_old}")
    print(f"{'='*90}")

    diffs = compute_diff(sm_conn, snap_new, snap_old)

    if not diffs:
        print(f"\n  No position changes detected between snapshots.")
        return

    # Separate entries from exits
    entries = [d for d in diffs if d["action"] in ("ENTER", "INCREASE")]
    exits = [d for d in diffs if d["action"] in ("EXIT", "DECREASE")]

    # Store diffs
    diff_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    now = datetime.now(timezone.utc).isoformat()

    # Get wallet usernames
    usernames = {}
    for row in sm_conn.execute("SELECT address, username FROM tracked_wallets").fetchall():
        usernames[row[0]] = row[1]

    for d in diffs:
        alerts_conn.execute("""
            INSERT INTO position_diffs
            (timestamp, diff_id, wallet, username, market_slug, question,
             outcome, action, size_change, new_size, price_at_diff, dollar_value)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (now, diff_id, d["wallet"], usernames.get(d["wallet"], "?"),
              d["slug"], (d["question"] or "")[:100], d["outcome"],
              d["action"], d["size_change"], d["new_size"],
              d["price"], d["dollar_value"]))
    alerts_conn.commit()

    # Display entries
    if entries:
        entries.sort(key=lambda x: -x["dollar_value"])
        print(f"\n  NEW ENTRIES / INCREASES ({len(entries)}):")
        print(f"  {'Wallet':<15s} {'Action':<8s} {'Market':<35s} "
              f"{'Side':<5s} {'Value':>10s} {'Price':>6s}")
        print(f"  {'-'*85}")

        for d in entries[:30]:
            uname = usernames.get(d["wallet"], d["wallet"][:12])[:14]
            print(f"  {uname:<15s} {d['action']:<8s} "
                  f"{(d['question'] or d['slug'])[:34]:<35s} "
                  f"{d['outcome']:<5s} ${d['dollar_value']:>9,.0f} "
                  f"{d['price']:>5.0%}")

    if exits:
        exits.sort(key=lambda x: -x["dollar_value"])
        print(f"\n  EXITS / DECREASES ({len(exits)}):")
        print(f"  {'Wallet':<15s} {'Action':<8s} {'Market':<35s} "
              f"{'Side':<5s} {'Value':>10s}")
        print(f"  {'-'*85}")

        for d in exits[:20]:
            uname = usernames.get(d["wallet"], d["wallet"][:12])[:14]
            print(f"  {uname:<15s} {d['action']:<8s} "
                  f"{(d['question'] or d['slug'])[:34]:<35s} "
                  f"{d['outcome']:<5s} ${d['dollar_value']:>9,.0f}")

    # Check convergence
    signals = detect_convergence(diffs)
    if signals:
        print(f"\n  CONVERGENCE SIGNALS:")
        print(f"  {'Market':<40s} {'Side':<5s} {'Wallets':>7s} "
              f"{'Capital':>10s} {'Score':>6s}")
        print(f"  {'-'*75}")

        for s in signals:
            print(f"  {(s['question'] or s['slug'])[:39]:<40s} "
                  f"{s['outcome']:<5s} {s['n_wallets']:>7d} "
                  f"${s['total_capital']:>9,.0f} {s['score']:>6.0f}")

            # Store signal
            signal_id = f"SIG_{diff_id}_{s['slug'][:20]}"
            alerts_conn.execute("""
                INSERT INTO convergence_signals
                (timestamp, signal_id, market_slug, question, outcome,
                 n_wallets, wallets, total_capital, avg_entry_price,
                 market_price, score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (now, signal_id, s["slug"], (s["question"] or "")[:100],
                  s["outcome"], s["n_wallets"],
                  json.dumps(s["wallets"][:10]),
                  s["total_capital"], s["avg_entry_price"],
                  s["avg_entry_price"], s["score"]))

        alerts_conn.commit()

        print(f"\n  To trade a signal:")
        print(f"  python -m engines.smart_money_alerts trade --signal \"SLUG\"")

    print(f"\n  Summary: {len(entries)} entries, {len(exits)} exits, "
          f"{len(signals)} convergence signals")
    print(f"{'='*90}")


def cmd_convergence(args):
    """Show all active convergence signals."""
    alerts_conn = init_alerts_db()
    min_wallets = getattr(args, "min", DEFAULT_MIN_CONVERGENCE)

    signals = alerts_conn.execute("""
        SELECT * FROM convergence_signals
        WHERE status='ACTIVE'
        ORDER BY score DESC
    """).fetchall()

    print(f"\n{'='*90}")
    print(f"  ACTIVE CONVERGENCE SIGNALS (min {min_wallets} wallets)")
    print(f"{'='*90}")

    if not signals:
        print(f"\n  No active signals. Run diff first:")
        print(f"  python -m engines.smart_money snapshot")
        print(f"  (wait 1+ hours)")
        print(f"  python -m engines.smart_money snapshot")
        print(f"  python -m engines.smart_money_alerts diff")
    else:
        print(f"\n  {'Time':<20s} {'Market':<35s} {'Side':<5s} "
              f"{'Wallets':>7s} {'$Capital':>10s} {'Score':>6s}")
        print(f"  {'-'*85}")

        for s in signals:
            if s[7] >= min_wallets:  # n_wallets
                print(f"  {s[1][:19]:<20s} {(s[4] or s[3])[:34]:<35s} "
                      f"{s[5]:<5s} {s[7]:>7d} "
                      f"${s[8]:>9,.0f} {s[11]:>6.0f}")

    print(f"{'='*90}")


def cmd_trade(args):
    """Execute a trade based on a convergence signal."""
    signal_slug = args.signal
    live = getattr(args, "live", False)
    size = getattr(args, "size", MAX_TRADE_SIZE)
    mode = "LIVE" if live else "PAPER"

    alerts_conn = init_alerts_db()

    # Find the signal
    signal = alerts_conn.execute("""
        SELECT * FROM convergence_signals
        WHERE market_slug LIKE ? AND status='ACTIVE'
        ORDER BY score DESC LIMIT 1
    """, (f"%{signal_slug}%",)).fetchone()

    if not signal:
        print(f"\n  No active signal matching: {signal_slug}")
        return

    print(f"\n{'='*70}")
    print(f"  SIGNAL TRADE -- {mode}")
    print(f"{'='*70}")
    print(f"  Market:   {signal[4]}")  # question
    print(f"  Side:     {signal[5]}")  # outcome
    print(f"  Wallets:  {signal[7]}")  # n_wallets
    print(f"  Capital:  ${signal[8]:,.0f}")  # total_capital
    print(f"  Score:    {signal[11]:.0f}")
    print(f"  Size:     ${size:.2f}")

    # Get current market price
    try:
        r = requests.get(f"{GAMMA_API}/markets", params={
            "slug": signal[3],  # market_slug
        }, timeout=10)
        markets = r.json()
        if markets:
            token_ids = json.loads(markets[0].get("clobTokenIds", "[]"))
            if token_ids:
                r = requests.get(f"{CLOB_API}/midpoint",
                                 params={"token_id": token_ids[0]}, timeout=5)
                data = r.json()
                current_price = float(
                    data.get("mid", 0.5) if isinstance(data, dict) else data)
                print(f"  Current:  {current_price:.1%}")
    except Exception:
        current_price = signal[10]  # market_price from signal

    now = datetime.now(timezone.utc).isoformat()
    signal_id = signal[2]  # signal_id

    if not live:
        # Paper trade
        shares = size / current_price if current_price > 0 else 0

        alerts_conn.execute("""
            INSERT INTO signal_trades
            (timestamp, signal_id, market_slug, outcome, side,
             size, price, cost, mode, status)
            VALUES (?, ?, ?, ?, 'BUY', ?, ?, ?, 'PAPER', 'FILLED')
        """, (now, signal_id, signal[3], signal[5],
              shares, current_price, size))
        alerts_conn.commit()

        print(f"\n  PAPER TRADE:")
        print(f"    BUY {shares:.1f} shares of {signal[5]} @ {current_price:.1%}")
        print(f"    Cost: ${size:.2f}")
        print(f"    Payout if correct: ${shares * 0.98:.2f}")
        print(f"    Potential profit: ${shares * 0.98 - size:.2f}")

    else:
        # Live trade
        pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
        if not pk:
            print(f"  No POLYMARKET_PRIVATE_KEY")
            return

        confirm = input(f"\n  Execute LIVE trade of ${size:.2f}? (yes/no): ")
        if confirm.lower() != "yes":
            print(f"  Cancelled.")
            return

        try:
            from py_clob_client.client import ClobClient
            client = ClobClient(CLOB_API, key=pk, chain_id=137)

            # Determine token ID based on outcome
            token_idx = 0 if signal[5] == "Yes" else 1
            token_id = token_ids[token_idx] if token_ids else ""

            order = client.create_and_post_order(
                token_id=token_id,
                side="BUY",
                size=size / current_price,
                price=round(min(current_price + 0.02, 0.99), 2),
            )

            order_id = order.get("orderID", order.get("id", ""))
            print(f"  Order submitted: {order_id}")

            alerts_conn.execute("""
                INSERT INTO signal_trades
                (timestamp, signal_id, market_slug, outcome, side,
                 size, price, cost, mode, order_id, status)
                VALUES (?, ?, ?, ?, 'BUY', ?, ?, ?, 'LIVE', ?, 'SUBMITTED')
            """, (now, signal_id, signal[3], signal[5],
                  size / current_price, current_price, size,
                  order_id))
            alerts_conn.commit()

        except Exception as e:
            print(f"  Trade failed: {e}")

    print(f"{'='*70}")


def cmd_performance(args):
    """Track how convergence signals perform after resolution."""
    alerts_conn = init_alerts_db()

    # Get all signals with trades
    trades = alerts_conn.execute("""
        SELECT st.*, cs.question, cs.outcome, cs.n_wallets, cs.score
        FROM signal_trades st
        JOIN convergence_signals cs ON st.signal_id = cs.signal_id
        ORDER BY st.timestamp DESC
    """).fetchall()

    print(f"\n{'='*70}")
    print(f"  SIGNAL PERFORMANCE TRACKER")
    print(f"{'='*70}")

    if not trades:
        print(f"\n  No trades yet. Run diff to find signals, then trade them.")
    else:
        total_cost = 0
        total_pnl = 0
        wins = 0
        total = len(trades)

        print(f"\n  {'Time':<20s} {'Market':<30s} {'Side':<5s} "
              f"{'Cost':>7s} {'P&L':>7s} {'Status'}")
        print(f"  {'-'*75}")

        for t in trades:
            cost = t[8] or 0  # cost
            pnl = t[13] or 0  # pnl
            total_cost += cost
            total_pnl += pnl
            if pnl > 0:
                wins += 1

            print(f"  {t[1][:19]:<20s} {(t[14] or t[3])[:29]:<30s} "
                  f"{(t[15] or t[4]):<5s} ${cost:>6.0f} "
                  f"${pnl:>+6.0f} {t[11]}")

        win_rate = wins / total * 100 if total > 0 else 0
        print(f"\n  Total trades: {total} | Win rate: {win_rate:.0f}%")
        print(f"  Total cost: ${total_cost:.0f} | Total P&L: ${total_pnl:+.0f}")

    # Also show untraded signals
    untraded = alerts_conn.execute("""
        SELECT * FROM convergence_signals
        WHERE status='ACTIVE' AND trade_id IS NULL
        ORDER BY score DESC
    """).fetchall()

    if untraded:
        print(f"\n  Untraded signals: {len(untraded)}")

    print(f"{'='*70}")


def cmd_monitor(args):
    """Continuous monitoring: snapshot -> diff -> alert."""
    interval = getattr(args, "interval", 3600)  # Default 1 hour

    print(f"\n{'='*70}")
    print(f"  SMART MONEY MONITOR")
    print(f"  Interval: {interval}s | Min convergence: {DEFAULT_MIN_CONVERGENCE}")
    print(f"  Press Ctrl+C to stop.")
    print(f"{'='*70}")

    cycle = 0
    while True:
        try:
            cycle += 1
            print(f"\n  -- Cycle {cycle} [{datetime.now().strftime('%H:%M:%S')}] --")

            # Take snapshot
            print(f"  Taking snapshot...")
            os.system(f"{sys.executable} -m engines.smart_money snapshot")

            time.sleep(5)

            # Run diff
            print(f"  Computing diff...")

            class FakeArgs:
                hours = None
            cmd_diff(FakeArgs())

            print(f"  Sleeping {interval}s...")
            time.sleep(interval)

        except KeyboardInterrupt:
            print(f"\n  Stopped after {cycle} cycles.")
            break
        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Smart Money Alerts")
    subs = parser.add_subparsers(dest="command")

    p_diff = subs.add_parser("diff", help="Show position changes")
    p_diff.add_argument("--hours", type=int, default=None)

    p_conv = subs.add_parser("convergence", help="Active convergence signals")
    p_conv.add_argument("--min", type=int, default=DEFAULT_MIN_CONVERGENCE)

    p_trade = subs.add_parser("trade", help="Trade a signal")
    p_trade.add_argument("--signal", required=True, help="Market slug to trade")
    p_trade.add_argument("--live", action="store_true")
    p_trade.add_argument("--size", type=float, default=MAX_TRADE_SIZE)

    subs.add_parser("performance", help="Signal performance tracking")

    p_mon = subs.add_parser("monitor", help="Continuous monitoring")
    p_mon.add_argument("--interval", type=int, default=3600)

    args = parser.parse_args()

    dispatch = {
        "diff": cmd_diff,
        "convergence": cmd_convergence,
        "trade": cmd_trade,
        "performance": cmd_performance,
        "monitor": cmd_monitor,
    }

    if args.command in dispatch:
        dispatch[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
