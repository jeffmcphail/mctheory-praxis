"""
engines/mev_executor.py — MEV Phase 2: Event-Driven Spike Executor

Automated execution engine that detects spikes in real-time and trades them
on the Polymarket CLOB. Works with the live collector's spike detection and
(eventually) the trained Stage 1/Stage 2 models.

Architecture:
    ┌──────────────────────────────────────────────────────────────┐
    │  DETECTION LAYER (from live_collector)                       │
    │  • Monitors price_snapshots for moves > threshold           │
    │  • Classifies event type via taxonomy                       │
    │  • Fires trigger when conditions met                        │
    └──────────────┬───────────────────────────────────────────────┘
                   │
    ┌──────────────▼───────────────────────────────────────────────┐
    │  DECISION LAYER                                              │
    │  Phase A (rule-based): Event type filter + threshold rules   │
    │  Phase B (ML): Stage 1 classifier (is this a genuine spike?) │
    └──────────────┬───────────────────────────────────────────────┘
                   │
    ┌──────────────▼───────────────────────────────────────────────┐
    │  SIZING LAYER                                                │
    │  Phase A: Fixed size per trade (e.g., $5)                    │
    │  Phase B (ML): Stage 2 peak proximity → dynamic sizing       │
    │    exposure = (1 - 2P) × max_size                            │
    │    P near 0 → full ride | P near 1 → reverse/fade           │
    └──────────────┬───────────────────────────────────────────────┘
                   │
    ┌──────────────▼───────────────────────────────────────────────┐
    │  EXECUTION LAYER                                             │
    │  • Posts limit orders on CLOB                                │
    │  • Monitors fills                                            │
    │  • Manages open positions                                    │
    │  • Exits on target/stop/timeout                              │
    └──────────────┬───────────────────────────────────────────────┘
                   │
    ┌──────────────▼───────────────────────────────────────────────┐
    │  RISK LAYER                                                  │
    │  • Max concurrent positions                                  │
    │  • Max daily loss                                            │
    │  • Per-trade max loss                                        │
    │  • Position timeout (auto-exit after N minutes)              │
    │  • Kill switch                                               │
    └──────────────────────────────────────────────────────────────┘

Modes:
    PAPER   — logs decisions but doesn't trade (default)
    LIVE    — executes real trades on CLOB
    BACKTEST — replays historical data through the engine

Usage:
    python -m engines.mev_executor run                     # Paper mode (default)
    python -m engines.mev_executor run --mode live         # LIVE trading
    python -m engines.mev_executor run --mode live --max-size 5  # $5 per trade
    python -m engines.mev_executor status                  # Show positions & P&L
    python -m engines.mev_executor history                 # Trade history
    python -m engines.mev_executor kill                    # Emergency close all
"""
import argparse
import json
import math
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from enum import Enum

import requests
from dotenv import load_dotenv
load_dotenv()

# ═══════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════

LIVE_DB = Path("data/live_collector.db")
EXECUTOR_DB = Path("data/mev_executor.db")
CLOB_API = "https://clob.polymarket.com"

# Detection thresholds
SPIKE_THRESHOLD_PCT = 8.0       # Minimum % move to trigger
SPIKE_WINDOW_MINS = 60          # Within this window
TRIGGER_COOLDOWN_MINS = 30      # Don't retrigger same market within N min

# Tradeable event types (from our analysis: sports & pop_culture revert)
TRADEABLE_TYPES = {"sports", "pop_culture"}
# Event types to NEVER trade (info-driven, don't revert)
EXCLUDED_TYPES = {"weather", "esports"}

# Risk parameters
DEFAULT_MAX_SIZE = 5.0          # $ per trade
MAX_CONCURRENT_POSITIONS = 3    # Max open positions at once
MAX_DAILY_LOSS = 25.0           # Stop trading after $25 daily loss
POSITION_TIMEOUT_MINS = 120     # Auto-exit after 2 hours
STOP_LOSS_PCT = 50.0            # Exit if position loses 50%+ of entry
TAKE_PROFIT_PCT = 30.0          # Exit if position gains 30%+ of entry

# Execution
SLIPPAGE_TOLERANCE = 0.03       # Max 3% slippage from mid price
ORDER_CHECK_INTERVAL = 5        # Check fills every 5 seconds
POLL_INTERVAL = 30              # Check for new spikes every 30 seconds


class Mode(Enum):
    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"


class Signal(Enum):
    FADE = "fade"       # Bet against the spike (mean reversion)
    RIDE = "ride"       # Bet with the spike (momentum)
    SKIP = "skip"       # Don't trade


# ═══════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════

def init_executor_db():
    """Initialize the executor database."""
    EXECUTOR_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(EXECUTOR_DB))
    conn.execute("PRAGMA journal_mode=WAL")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slug TEXT NOT NULL,
            question TEXT,
            event_type TEXT,

            -- Detection
            detected_at TEXT,
            trigger_price REAL,
            baseline_price REAL,
            spike_pct REAL,
            spike_direction TEXT,

            -- Decision
            signal TEXT,
            signal_reason TEXT,
            confidence REAL,

            -- Execution
            mode TEXT,
            side TEXT,
            token_id TEXT,
            order_id TEXT,
            entry_price REAL,
            size_shares REAL,
            cost_usd REAL,

            -- Position management
            status TEXT DEFAULT 'pending',
            exit_price REAL,
            exit_reason TEXT,
            exit_at TEXT,
            pnl_usd REAL,
            pnl_pct REAL,

            -- Metadata
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_pnl (
            date TEXT PRIMARY KEY,
            trades INTEGER DEFAULT 0,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            gross_pnl REAL DEFAULT 0,
            fees_paid REAL DEFAULT 0,
            net_pnl REAL DEFAULT 0
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS kill_switch (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            active INTEGER DEFAULT 0,
            reason TEXT,
            activated_at TEXT
        )
    """)
    conn.execute("INSERT OR IGNORE INTO kill_switch (id, active) VALUES (1, 0)")

    conn.commit()
    return conn


# ═══════════════════════════════════════════════════════
# DETECTION
# ═══════════════════════════════════════════════════════

def get_recent_spikes(conn_live, threshold=SPIKE_THRESHOLD_PCT,
                      window_mins=SPIKE_WINDOW_MINS):
    """Scan live collector data for recent spikes.

    Returns list of spike dicts with market info.
    """
    now = int(time.time())
    window_start = now - (window_mins * 60)
    lookback = now - (TRIGGER_COOLDOWN_MINS * 60)

    # Get all active markets with recent price data
    markets = conn_live.execute("""
        SELECT slug, question, event_type, yes_token, no_token
        FROM tracked_markets WHERE active=1
    """).fetchall()

    spikes = []

    for market in markets:
        slug = market["slug"]
        yes_token = market["yes_token"]
        no_token = market["no_token"]

        # Get price history in window
        prices = conn_live.execute("""
            SELECT timestamp, yes_mid FROM price_snapshots
            WHERE slug=? AND timestamp >= ?
            ORDER BY timestamp ASC
        """, (slug, window_start)).fetchall()

        if len(prices) < 5:
            continue

        # Find max move within window
        first_price = prices[0]["yes_mid"]
        last_price = prices[-1]["yes_mid"]

        if first_price <= 0.05 or first_price >= 0.95:
            continue

        move_pct = (last_price - first_price) / first_price * 100

        if abs(move_pct) >= threshold:
            # Find baseline (average of first 5 prices)
            baseline_prices = [p["yes_mid"] for p in prices[:5]]
            baseline = sum(baseline_prices) / len(baseline_prices)

            direction = "UP" if move_pct > 0 else "DOWN"

            spikes.append({
                "slug": slug,
                "question": market["question"] or slug,
                "event_type": market["event_type"] or "unknown",
                "yes_token": yes_token,
                "no_token": no_token,
                "baseline_price": baseline,
                "trigger_price": last_price,
                "spike_pct": move_pct,
                "direction": direction,
                "timestamp": now,
                "n_prices": len(prices),
            })

    return spikes


# ═══════════════════════════════════════════════════════
# DECISION
# ═══════════════════════════════════════════════════════

def decide_signal(spike, conn_exec):
    """Decide whether and how to trade a detected spike.

    Phase A: Rule-based (event type + threshold)
    Phase B: ML-based (Stage 1 classifier) — TODO after model training

    Returns:
        (Signal, confidence, reason)
    """
    slug = spike["slug"]
    event_type = spike["event_type"]
    direction = spike["direction"]
    move_pct = abs(spike["spike_pct"])
    trigger_price = spike["trigger_price"]

    # ── Kill switch check ──
    kill = conn_exec.execute(
        "SELECT active FROM kill_switch WHERE id=1").fetchone()
    if kill and kill["active"]:
        return Signal.SKIP, 0, "Kill switch active"

    # ── Check daily loss limit ──
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    daily = conn_exec.execute(
        "SELECT net_pnl FROM daily_pnl WHERE date=?", (today,)).fetchone()
    if daily and daily["net_pnl"] <= -MAX_DAILY_LOSS:
        return Signal.SKIP, 0, f"Daily loss limit reached (${daily['net_pnl']:.2f})"

    # ── Check concurrent positions ──
    open_count = conn_exec.execute(
        "SELECT COUNT(*) FROM trades WHERE status='open'").fetchone()[0]
    if open_count >= MAX_CONCURRENT_POSITIONS:
        return Signal.SKIP, 0, f"Max concurrent positions ({MAX_CONCURRENT_POSITIONS})"

    # ── Check cooldown ──
    recent = conn_exec.execute("""
        SELECT id FROM trades
        WHERE slug=? AND created_at > datetime('now', ?)
    """, (slug, f"-{TRIGGER_COOLDOWN_MINS} minutes")).fetchone()
    if recent:
        return Signal.SKIP, 0, f"Cooldown ({TRIGGER_COOLDOWN_MINS}m) for {slug}"

    # ── Event type filter ──
    if event_type in EXCLUDED_TYPES:
        return Signal.SKIP, 0, f"Excluded event type: {event_type}"

    if event_type not in TRADEABLE_TYPES and event_type != "unknown":
        return Signal.SKIP, 0.3, f"Untested event type: {event_type}"

    # ── Price boundary check ──
    # Don't trade if price is already near 0 or 1 (resolved or nearly resolved)
    if trigger_price < 0.08 or trigger_price > 0.92:
        return Signal.SKIP, 0, f"Price too extreme ({trigger_price:.3f})"

    # ── FADE signal (bet against the spike) ──
    # Our analysis shows sports/pop_culture spike 72%/64% reversion
    # So we fade: buy the OPPOSITE side of the spike
    confidence = 0.5

    # Higher confidence for larger spikes (more likely to revert)
    if move_pct >= 20:
        confidence = 0.7
    elif move_pct >= 15:
        confidence = 0.65
    elif move_pct >= 10:
        confidence = 0.6

    # Higher confidence for known-tradeable types
    if event_type == "sports":
        confidence += 0.1  # 72% reversion
    elif event_type == "pop_culture":
        confidence += 0.05  # 64% reversion

    confidence = min(confidence, 0.9)

    reason = (f"FADE {direction} spike: {event_type} {move_pct:.1f}% move, "
              f"expecting mean reversion")

    return Signal.FADE, confidence, reason


def compute_trade_params(spike, signal, max_size):
    """Compute the trade parameters (side, price, size).

    FADE strategy: bet AGAINST the spike direction.
    - If spike is UP (price went up), we SELL YES / BUY NO → price should come back down
    - If spike is DOWN (price went down), we BUY YES → price should come back up
    """
    direction = spike["direction"]
    trigger_price = spike["trigger_price"]

    if signal == Signal.FADE:
        if direction == "UP":
            # Price spiked UP → fade by buying NO (or selling YES)
            # Buying NO is cleaner — we pay (1 - trigger_price) for NO
            side = "BUY"
            token_id = spike["no_token"]
            entry_price = round(1 - trigger_price, 2)  # NO price ≈ 1 - YES price
            # Ensure minimum tick
            entry_price = max(entry_price, 0.01)
        else:
            # Price spiked DOWN → fade by buying YES
            side = "BUY"
            token_id = spike["yes_token"]
            entry_price = trigger_price

    elif signal == Signal.RIDE:
        if direction == "UP":
            side = "BUY"
            token_id = spike["yes_token"]
            entry_price = trigger_price
        else:
            side = "BUY"
            token_id = spike["no_token"]
            entry_price = round(1 - trigger_price, 2)
            entry_price = max(entry_price, 0.01)
    else:
        return None

    # Size: max_size USD → shares at entry_price
    if entry_price <= 0:
        return None

    size_shares = max_size / entry_price
    cost_usd = max_size

    # Add slippage buffer to limit price
    limit_price = round(entry_price + SLIPPAGE_TOLERANCE, 2)
    limit_price = min(limit_price, 0.99)

    return {
        "side": side,
        "token_id": token_id,
        "entry_price": entry_price,
        "limit_price": limit_price,
        "size_shares": round(size_shares, 1),
        "cost_usd": cost_usd,
    }


# ═══════════════════════════════════════════════════════
# EXECUTION
# ═══════════════════════════════════════════════════════

def execute_trade(trade_params, mode=Mode.PAPER):
    """Execute a trade on the CLOB.

    Returns:
        order_id or None
    """
    if mode == Mode.PAPER:
        # Simulate fill at entry price
        return f"PAPER-{int(time.time())}"

    if mode == Mode.LIVE:
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import OrderArgs
            from py_clob_client.order_builder.constants import BUY, SELL

            pk = os.getenv("POLYMARKET_PRIVATE_KEY", "")
            if not pk:
                print("  ❌ No POLYMARKET_PRIVATE_KEY")
                return None

            clob = ClobClient("https://clob.polymarket.com", key=pk, chain_id=137)
            try:
                clob.set_api_creds(clob.create_or_derive_api_creds())
            except Exception:
                clob.set_api_creds(clob.derive_api_key())

            side_const = BUY if trade_params["side"] == "BUY" else SELL

            order_args = OrderArgs(
                price=trade_params["limit_price"],
                size=trade_params["size_shares"],
                side=side_const,
                token_id=trade_params["token_id"],
            )

            result = clob.create_and_post_order(order_args)

            if isinstance(result, dict) and result.get("success"):
                return result.get("orderID", "unknown")
            else:
                print(f"  ❌ Order failed: {result}")
                return None

        except Exception as e:
            print(f"  ❌ Execution error: {e}")
            return None

    return None


def check_position_exit(trade, conn_live, conn_exec, mode=Mode.PAPER):
    """Check if an open position should be exited.

    Exit conditions:
    1. Take profit hit
    2. Stop loss hit
    3. Position timeout
    4. Kill switch activated

    Returns:
        (should_exit, reason, current_price)
    """
    slug = trade["slug"]
    entry_price = trade["entry_price"]
    created_at = trade["created_at"]

    # Get current price
    latest = conn_live.execute("""
        SELECT yes_mid FROM price_snapshots
        WHERE slug=? ORDER BY timestamp DESC LIMIT 1
    """, (slug,)).fetchone()

    if not latest:
        return False, None, None

    current_yes = latest["yes_mid"]

    # Determine our position's current value
    # If we bought NO (fading UP spike), our value = 1 - current_yes
    # If we bought YES (fading DOWN spike), our value = current_yes
    if trade["side"] == "BUY" and trade["token_id"] == trade.get("no_token", ""):
        current_value = 1 - current_yes
    else:
        current_value = current_yes

    # P&L calculation
    if entry_price > 0:
        pnl_pct = (current_value - entry_price) / entry_price * 100
    else:
        pnl_pct = 0

    # Check kill switch
    kill = conn_exec.execute(
        "SELECT active FROM kill_switch WHERE id=1").fetchone()
    if kill and kill["active"]:
        return True, "kill_switch", current_value

    # Check take profit
    if pnl_pct >= TAKE_PROFIT_PCT:
        return True, f"take_profit ({pnl_pct:+.1f}%)", current_value

    # Check stop loss
    if pnl_pct <= -STOP_LOSS_PCT:
        return True, f"stop_loss ({pnl_pct:+.1f}%)", current_value

    # Check timeout
    try:
        entry_time = datetime.fromisoformat(created_at)
        elapsed = datetime.now(timezone.utc) - entry_time.replace(tzinfo=timezone.utc)
        if elapsed > timedelta(minutes=POSITION_TIMEOUT_MINS):
            return True, f"timeout ({POSITION_TIMEOUT_MINS}m)", current_value
    except Exception:
        pass

    return False, None, current_value


# ═══════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════

def cmd_run(args):
    """Main executor loop."""
    mode = Mode(getattr(args, "mode", "paper"))
    max_size = getattr(args, "max_size", DEFAULT_MAX_SIZE)

    if not LIVE_DB.exists():
        print("  ❌ No live collector database. Start the collector first.")
        return

    conn_live = sqlite3.connect(str(LIVE_DB))
    conn_live.row_factory = sqlite3.Row
    conn_exec = init_executor_db()
    conn_exec.row_factory = sqlite3.Row

    mode_label = "🔴 LIVE" if mode == Mode.LIVE else "📝 PAPER"

    print(f"\n{'='*90}")
    print(f"  MEV EXECUTOR — {mode_label}")
    print(f"  Max size: ${max_size:.2f}/trade | "
          f"Max concurrent: {MAX_CONCURRENT_POSITIONS} | "
          f"Max daily loss: ${MAX_DAILY_LOSS:.2f}")
    print(f"  Tradeable types: {', '.join(sorted(TRADEABLE_TYPES))}")
    print(f"  Spike threshold: {SPIKE_THRESHOLD_PCT}% in {SPIKE_WINDOW_MINS}m")
    print(f"  Take profit: {TAKE_PROFIT_PCT}% | "
          f"Stop loss: {STOP_LOSS_PCT}% | "
          f"Timeout: {POSITION_TIMEOUT_MINS}m")
    print(f"  Press Ctrl+C to stop.")
    print(f"{'='*90}")

    if mode == Mode.LIVE:
        print(f"\n  ⚠️  LIVE MODE — Real money will be used!")
        print(f"  Type 'yes' to confirm: ", end="")
        if input().strip().lower() != "yes":
            print("  Aborted.")
            return

    cycle = 0
    total_trades = 0

    while True:
        try:
            cycle += 1
            now_str = datetime.now().strftime("%H:%M:%S")

            # ── Check open positions for exit ──
            open_trades = conn_exec.execute(
                "SELECT * FROM trades WHERE status='open'").fetchall()

            for trade in open_trades:
                should_exit, reason, current_value = check_position_exit(
                    dict(trade), conn_live, conn_exec, mode)

                if should_exit:
                    pnl = 0
                    if trade["entry_price"] and current_value:
                        pnl = (current_value - trade["entry_price"]) * trade["size_shares"]
                        # Subtract 2% fee on exit
                        pnl -= trade["cost_usd"] * 0.02

                    pnl_pct = 0
                    if trade["entry_price"] > 0:
                        pnl_pct = (current_value - trade["entry_price"]) / trade["entry_price"] * 100

                    conn_exec.execute("""
                        UPDATE trades SET
                            status='closed', exit_price=?, exit_reason=?,
                            exit_at=?, pnl_usd=?, pnl_pct=?,
                            updated_at=datetime('now')
                        WHERE id=?
                    """, (current_value, reason,
                          datetime.now(timezone.utc).isoformat(),
                          pnl, pnl_pct, trade["id"]))
                    conn_exec.commit()

                    icon = "💰" if pnl > 0 else "💸"
                    print(f"  {now_str} {icon} EXIT [{trade['slug'][:30]}] "
                          f"{reason} | P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)")

                    # Update daily P&L
                    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    conn_exec.execute("""
                        INSERT INTO daily_pnl (date, trades, wins, losses, net_pnl)
                        VALUES (?, 1, ?, ?, ?)
                        ON CONFLICT(date) DO UPDATE SET
                            trades = trades + 1,
                            wins = wins + ?,
                            losses = losses + ?,
                            net_pnl = net_pnl + ?
                    """, (today,
                          1 if pnl > 0 else 0, 1 if pnl <= 0 else 0, pnl,
                          1 if pnl > 0 else 0, 1 if pnl <= 0 else 0, pnl))
                    conn_exec.commit()

            # ── Scan for new spikes ──
            spikes = get_recent_spikes(conn_live)

            for spike in spikes:
                signal, confidence, reason = decide_signal(spike, conn_exec)

                if signal == Signal.SKIP:
                    if cycle <= 3:  # Only show skips on first few cycles
                        print(f"  {now_str} ⏭️  SKIP [{spike['slug'][:30]}] {reason}")
                    continue

                # Compute trade params
                params = compute_trade_params(spike, signal, max_size)
                if not params:
                    continue

                # Execute
                order_id = execute_trade(params, mode)

                if order_id:
                    total_trades += 1
                    conn_exec.execute("""
                        INSERT INTO trades
                        (slug, question, event_type, detected_at,
                         trigger_price, baseline_price, spike_pct, spike_direction,
                         signal, signal_reason, confidence,
                         mode, side, token_id, order_id,
                         entry_price, size_shares, cost_usd, status)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')
                    """, (
                        spike["slug"], spike["question"][:100],
                        spike["event_type"],
                        datetime.now(timezone.utc).isoformat(),
                        spike["trigger_price"], spike["baseline_price"],
                        spike["spike_pct"], spike["direction"],
                        signal.value, reason, confidence,
                        mode.value, params["side"], params["token_id"],
                        order_id, params["entry_price"],
                        params["size_shares"], params["cost_usd"],
                    ))
                    conn_exec.commit()

                    direction_icon = "📈" if spike["direction"] == "UP" else "📉"
                    print(f"  {now_str} {direction_icon} {signal.value.upper()} "
                          f"[{spike['slug'][:30]}] "
                          f"{spike['spike_pct']:+.1f}% | "
                          f"{params['side']} {params['size_shares']:.1f} "
                          f"@ {params['entry_price']:.3f} "
                          f"(${params['cost_usd']:.2f}) "
                          f"| conf={confidence:.0%} [{spike['event_type']}]")

            # ── Status line (every 10 cycles) ──
            if cycle % 10 == 0:
                open_count = conn_exec.execute(
                    "SELECT COUNT(*) FROM trades WHERE status='open'").fetchone()[0]
                total_pnl = conn_exec.execute(
                    "SELECT COALESCE(SUM(pnl_usd), 0) FROM trades WHERE status='closed'"
                ).fetchone()[0]

                print(f"  {now_str} 📊 Cycle {cycle} | "
                      f"Open: {open_count} | "
                      f"Total trades: {total_trades} | "
                      f"Closed P&L: ${total_pnl:+.2f}")

            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            print(f"\n\n  Stopped after {cycle} cycles, {total_trades} trades.")
            # Show summary
            cmd_status_internal(conn_exec)
            break
        except Exception as e:
            print(f"  ⚠️ Error: {e}")
            time.sleep(POLL_INTERVAL)

    conn_live.close()
    conn_exec.close()


# ═══════════════════════════════════════════════════════
# STATUS & HISTORY
# ═══════════════════════════════════════════════════════

def cmd_status_internal(conn):
    """Print status summary (internal, conn already open)."""
    open_trades = conn.execute(
        "SELECT * FROM trades WHERE status='open' ORDER BY created_at DESC"
    ).fetchall()

    closed_trades = conn.execute(
        "SELECT * FROM trades WHERE status='closed' ORDER BY exit_at DESC LIMIT 20"
    ).fetchall()

    total_pnl = conn.execute(
        "SELECT COALESCE(SUM(pnl_usd), 0) FROM trades WHERE status='closed'"
    ).fetchone()[0]

    total_trades = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    wins = conn.execute(
        "SELECT COUNT(*) FROM trades WHERE status='closed' AND pnl_usd > 0"
    ).fetchone()[0]
    losses = conn.execute(
        "SELECT COUNT(*) FROM trades WHERE status='closed' AND pnl_usd <= 0"
    ).fetchone()[0]

    print(f"\n{'─'*90}")
    print(f"  EXECUTOR STATUS")
    print(f"{'─'*90}")
    print(f"  Total trades:  {total_trades}")
    print(f"  Open:          {len(open_trades)}")
    print(f"  Closed:        {wins + losses} (W:{wins} L:{losses})")
    if wins + losses > 0:
        print(f"  Win rate:      {wins/(wins+losses):.0%}")
    print(f"  Total P&L:     ${total_pnl:+.2f}")

    if open_trades:
        print(f"\n  Open Positions:")
        print(f"  {'Market':<35s} {'Signal':<6s} {'Entry':>6s} "
              f"{'Size':>6s} {'Cost':>7s} {'Type':<12s}")
        print(f"  {'─'*80}")
        for t in open_trades:
            print(f"  {t['slug'][:34]:<35s} {t['signal']:<6s} "
                  f"{t['entry_price']:>6.3f} {t['size_shares']:>6.1f} "
                  f"${t['cost_usd']:>6.2f} {t['event_type']:<12s}")

    if closed_trades:
        print(f"\n  Recent Closed:")
        print(f"  {'Market':<30s} {'Signal':<6s} {'Entry':>6s} "
              f"{'Exit':>6s} {'P&L':>8s} {'Reason':<20s}")
        print(f"  {'─'*85}")
        for t in closed_trades[:10]:
            pnl = t["pnl_usd"] or 0
            icon = "✅" if pnl > 0 else "❌"
            print(f"  {t['slug'][:29]:<30s} {t['signal']:<6s} "
                  f"{t['entry_price']:>6.3f} {(t['exit_price'] or 0):>6.3f} "
                  f"${pnl:>+7.2f} {icon} {(t['exit_reason'] or '')[:20]:<20s}")

    print(f"\n{'─'*90}")


def cmd_status(args):
    """Show current executor status."""
    if not EXECUTOR_DB.exists():
        print("  No executor database. Run the executor first.")
        return
    conn = sqlite3.connect(str(EXECUTOR_DB))
    conn.row_factory = sqlite3.Row
    cmd_status_internal(conn)
    conn.close()


def cmd_history(args):
    """Show full trade history."""
    if not EXECUTOR_DB.exists():
        print("  No executor database.")
        return

    conn = sqlite3.connect(str(EXECUTOR_DB))
    conn.row_factory = sqlite3.Row

    trades = conn.execute(
        "SELECT * FROM trades ORDER BY created_at DESC").fetchall()

    print(f"\n{'='*100}")
    print(f"  TRADE HISTORY ({len(trades)} trades)")
    print(f"{'='*100}")

    print(f"  {'#':>3s} {'Time':<20s} {'Market':<30s} {'Type':<12s} "
          f"{'Signal':<6s} {'Spike':>7s} {'Entry':>6s} {'Exit':>6s} "
          f"{'P&L':>8s} {'Status':<8s}")
    print(f"  {'─'*110}")

    for i, t in enumerate(trades):
        pnl = t["pnl_usd"] or 0
        pnl_str = f"${pnl:>+7.2f}" if t["status"] == "closed" else "   open"
        print(f"  {i+1:>3d} {(t['detected_at'] or '')[:19]:<20s} "
              f"{t['slug'][:29]:<30s} {(t['event_type'] or '?'):<12s} "
              f"{t['signal']:<6s} {(t['spike_pct'] or 0):>+6.1f}% "
              f"{(t['entry_price'] or 0):>6.3f} {(t['exit_price'] or 0):>6.3f} "
              f"{pnl_str} {t['status']:<8s}")

    conn.close()
    print(f"\n{'='*100}")


def cmd_kill(args):
    """Emergency: activate kill switch, close all positions."""
    conn = init_executor_db()
    conn.row_factory = sqlite3.Row

    reason = getattr(args, "reason", "Manual kill switch")

    conn.execute("""
        UPDATE kill_switch SET active=1, reason=?, activated_at=datetime('now')
        WHERE id=1
    """, (reason,))

    # Mark all open positions as closed
    open_trades = conn.execute(
        "SELECT id FROM trades WHERE status='open'").fetchall()

    for t in open_trades:
        conn.execute("""
            UPDATE trades SET status='killed', exit_reason='kill_switch',
            exit_at=datetime('now'), updated_at=datetime('now')
            WHERE id=?
        """, (t["id"],))

    conn.commit()
    print(f"\n  🛑 KILL SWITCH ACTIVATED")
    print(f"  {len(open_trades)} positions marked as killed")
    print(f"  Reason: {reason}")
    print(f"\n  To deactivate:")
    print(f"    python -c \"import sqlite3; "
          f"c=sqlite3.connect('data/mev_executor.db'); "
          f"c.execute('UPDATE kill_switch SET active=0 WHERE id=1'); "
          f"c.commit(); print('Kill switch deactivated')\"")

    conn.close()


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="MEV Executor — Phase 2")
    subs = parser.add_subparsers(dest="command")

    p_run = subs.add_parser("run", help="Start the executor")
    p_run.add_argument("--mode", choices=["paper", "live"], default="paper")
    p_run.add_argument("--max-size", type=float, default=DEFAULT_MAX_SIZE,
                        help=f"Max $ per trade (default: {DEFAULT_MAX_SIZE})")

    subs.add_parser("status", help="Show current status")
    subs.add_parser("history", help="Show trade history")

    p_kill = subs.add_parser("kill", help="Emergency kill switch")
    p_kill.add_argument("--reason", default="Manual kill switch")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "history":
        cmd_history(args)
    elif args.command == "kill":
        cmd_kill(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
