"""
engines/btc_momentum.py — 5-Minute BTC Momentum Bot

Implements the "4-minute rule": Watch 4 minutes of a 5-minute Polymarket
BTC market. If price pushes directionally without reversal, bet on
continuation in the final minute.

The edge: Polymarket's 5-min BTC markets don't have enough liquidity
or time for mean-reversion to fully reprice momentum. By minute 4,
the direction is baked in. Minute 5 is follow-through.

Architecture:
    1. DISCOVER — Find active 5-min BTC Up/Down markets
    2. COLLECT  — Stream BTC price + market odds every 10 seconds
    3. SIGNAL   — Detect directional momentum at trigger threshold
    4. EXECUTE  — Place continuation bet in final window
    5. ANALYZE  — Backtest trigger thresholds to optimize

Usage:
    python -m engines.btc_momentum discover          # Find active 5-min markets
    python -m engines.btc_momentum collect            # Start collecting data
    python -m engines.btc_momentum collect --minutes 60  # Collect for 60 min
    python -m engines.btc_momentum backtest           # Analyze collected data
    python -m engines.btc_momentum backtest --trigger 3.5  # Test 3.5 min trigger
"""
import argparse
import json
import os
import sqlite3
import time
from datetime import datetime, timezone

import requests

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

DB_PATH = "data/btc_momentum.db"


def init_db():
    """Initialize SQLite database for price collection."""
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS collections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            unix_ts INTEGER NOT NULL,
            btc_price REAL,
            market_id TEXT,
            market_question TEXT,
            market_end_ts INTEGER,
            up_price REAL,
            down_price REAL,
            up_token TEXT,
            down_token TEXT,
            seconds_remaining INTEGER,
            source TEXT DEFAULT 'live'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rounds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_id TEXT,
            start_ts INTEGER,
            end_ts INTEGER,
            open_btc REAL,
            close_btc REAL,
            btc_direction TEXT,
            up_price_at_trigger REAL,
            down_price_at_trigger REAL,
            trigger_seconds REAL,
            signal TEXT,
            outcome TEXT,
            profit REAL
        )
    """)
    conn.commit()
    return conn


def get_btc_price():
    """Get current BTC price from CoinGecko (free, no key)."""
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": "bitcoin", "vs_currencies": "usd"},
            timeout=5
        )
        if r.ok:
            return r.json().get("bitcoin", {}).get("usd", 0)
    except Exception:
        pass

    # Fallback: Binance
    try:
        r = requests.get("https://api.binance.com/api/v3/ticker/price",
                         params={"symbol": "BTCUSDT"}, timeout=5)
        if r.ok:
            return float(r.json().get("price", 0))
    except Exception:
        pass

    return 0


def discover_5min_markets():
    """Find active 5-minute BTC Up/Down markets on Polymarket.
    
    These markets use deterministic slugs based on window timestamps:
    btc-updown-5m-{window_start_unix_timestamp}
    """
    print(f"\n{'='*90}")
    print(f"  DISCOVERING 5-MINUTE BTC MARKETS")
    print(f"{'='*90}")

    now = int(time.time())
    btc = get_btc_price()
    print(f"\n  Current BTC: ${btc:,.2f}")
    print(f"  Current time: {datetime.fromtimestamp(now, tz=timezone.utc).strftime('%H:%M:%S UTC')}")

    # Calculate current and nearby window timestamps
    current_window = now - (now % 300)
    secs_into_window = now % 300
    secs_remaining = 300 - secs_into_window

    print(f"  Current window: {current_window} ({secs_into_window}s in, {secs_remaining}s left)")

    # Try current and nearby windows
    found = []
    slugs_to_try = []
    for offset in [-2, -1, 0, 1, 2]:
        ts = current_window + (offset * 300)
        slugs_to_try.append(f"btc-updown-5m-{ts}")

    # Also try event-level slugs
    event_slugs = [
        "bitcoin-up-or-down-5-minutes",
        "btc-5-minute-up-or-down",
        "bitcoin-5-min",
    ]

    print(f"\n  Testing deterministic slugs...")
    for slug in slugs_to_try:
        try:
            r = requests.get(f"{GAMMA_API}/markets", params={"slug": slug}, timeout=5)
            markets = r.json()
            if markets and isinstance(markets, list) and len(markets) > 0:
                m = markets[0]
                prices = json.loads(m.get("outcomePrices", "[0,0]"))
                tokens = json.loads(m.get("clobTokenIds", "[]"))
                vol = float(m.get("volume", 0))
                accepting = m.get("acceptingOrders", False)
                status = "✅ ACTIVE" if accepting else "⏸ CLOSED"
                print(f"  {status} {slug}  UP={float(prices[0]):.1%} DN={float(prices[1]):.1%}  "
                      f"Vol=${vol:,.0f}")
                found.append({
                    "slug": slug,
                    "question": m.get("question", ""),
                    "condition_id": m.get("conditionId", ""),
                    "outcome_prices": m.get("outcomePrices", ""),
                    "clob_token_ids": m.get("clobTokenIds", ""),
                    "volume": vol,
                    "accepting_orders": accepting,
                    "end_date": m.get("endDate", ""),
                })
            else:
                print(f"  ❌ {slug} — not found")
        except Exception as e:
            print(f"  ❌ {slug} — {e}")
        time.sleep(0.2)

    # Try event slugs
    print(f"\n  Testing event slugs...")
    for slug in event_slugs:
        try:
            r = requests.get(f"{GAMMA_API}/events", params={"slug": slug}, timeout=5)
            events = r.json()
            if events:
                event = events[0]
                print(f"  ✅ Event: {event.get('title', '?')} ({len(event.get('markets', []))} markets)")
                for m in event.get("markets", [])[:3]:
                    print(f"     {m.get('question', '')[:60]}  Vol=${float(m.get('volume', 0)):,.0f}")
                break
            else:
                print(f"  ❌ {slug} — not found")
        except Exception:
            print(f"  ❌ {slug} — error")
        time.sleep(0.2)

    # Try searching Gamma for any market with "btc" and "5" in the question
    print(f"\n  Searching Gamma API for BTC up/down markets...")
    try:
        # The crypto page shows categories: 5 Min, 15 Min, 1 Hour etc.
        # Let's fetch from the crypto section
        for tag in ["crypto"]:
            r = requests.get(f"{GAMMA_API}/events", params={
                "tag_slug": tag, "limit": "100", "active": "true",
                "closed": "false", "order": "volume", "ascending": "false",
            }, timeout=10)
            for event in r.json():
                title = event.get("title", "").lower()
                for m in event.get("markets", []):
                    q = m.get("question", "").lower()
                    slug = m.get("slug", "").lower()
                    if (("up or down" in q or "updown" in slug or "up-or-down" in slug)
                            and ("5 min" in q or "5m" in slug or "5-min" in q)):
                        vol = float(m.get("volume", 0))
                        if vol > 0:
                            print(f"  ✅ Found: {m.get('question', '')[:65]}")
                            print(f"     Slug: {m.get('slug', '')}")
                            print(f"     Vol: ${vol:,.0f}")
                            found.append({
                                "slug": m.get("slug", ""),
                                "question": m.get("question", ""),
                                "condition_id": m.get("conditionId", ""),
                                "outcome_prices": m.get("outcomePrices", ""),
                                "clob_token_ids": m.get("clobTokenIds", ""),
                                "volume": vol,
                                "accepting_orders": m.get("acceptingOrders", False),
                                "end_date": m.get("endDate", ""),
                            })
    except Exception as e:
        print(f"  Error: {e}")

    found.sort(key=lambda x: x["volume"], reverse=True)

    # Save results
    os.makedirs("data", exist_ok=True)
    with open("data/btc_5min_markets.json", "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "markets_found": len(found),
            "btc_price": btc,
            "current_window": current_window,
            "markets": found[:50],
        }, f, indent=2, default=str)

    if found:
        active = [m for m in found if m.get("accepting_orders")]
        print(f"\n  Found {len(found)} markets ({len(active)} currently accepting orders)")
    else:
        print(f"\n  No markets found. The 5-min BTC markets may use a different slug pattern.")
        print(f"  Check: https://polymarket.com/crypto/5M")

    print(f"\n  Saved: data/btc_5min_markets.json")
    print(f"{'='*90}")

    return found


def cmd_collect(args):
    """Collect BTC price + market odds data for backtesting.
    
    Uses the CLOB API for real-time pricing:
    - get_midpoint(token_id) for live mid-market price
    - get_price(token_id, side) for executable BUY/SELL prices
    - Gamma API only for token_id discovery (once per window)
    """
    minutes = getattr(args, "minutes", 30)
    interval = getattr(args, "interval", 10)

    print(f"\n{'='*90}")
    print(f"  BTC MOMENTUM DATA COLLECTOR (CLOB Real-Time Pricing)")
    print(f"  Duration: {minutes} minutes  |  Interval: {interval}s")
    print(f"{'='*90}")

    # Read-only CLOB client — no auth needed for price queries
    from py_clob_client.client import ClobClient as ReadClobClient
    clob = ReadClobClient("https://clob.polymarket.com")

    conn = init_db()

    # Add columns for CLOB data if not exist
    try:
        conn.execute("ALTER TABLE collections ADD COLUMN up_mid REAL")
        conn.execute("ALTER TABLE collections ADD COLUMN up_buy REAL")
        conn.execute("ALTER TABLE collections ADD COLUMN up_sell REAL")
        conn.execute("ALTER TABLE collections ADD COLUMN spread REAL")
        conn.commit()
    except Exception:
        pass  # Columns already exist

    total_samples = int(minutes * 60 / interval)
    last_window = None
    up_token = down_token = ""

    print(f"\n  Collecting {total_samples} samples across ~{minutes // 5} windows...")
    print(f"  {'Time':<10s} {'Slug':>8s} {'Left':>5s} {'BTC':>10s} "
          f"{'Mid':>6s} {'Buy':>6s} {'Sell':>6s} {'Sprd':>6s} {'Vol':>8s}")
    print(f"  {'─'*78}")

    for i in range(total_samples):
        try:
            now = int(time.time())
            now_dt = datetime.fromtimestamp(now, tz=timezone.utc)

            # Calculate current 5-min window
            window_ts = now - (now % 300)
            secs_remaining = 300 - (now % 300)
            slug = f"btc-updown-5m-{window_ts}"

            # Get BTC price
            btc = get_btc_price()

            # When window changes, fetch token IDs from Gamma
            if window_ts != last_window:
                if last_window is not None:
                    print(f"  {'─'*78}")
                print(f"  *** New window: {slug} ***")
                last_window = window_ts
                up_token = down_token = ""
                try:
                    r = requests.get(f"{GAMMA_API}/markets",
                                     params={"slug": slug}, timeout=5)
                    markets = r.json()
                    if markets and isinstance(markets, list) and len(markets) > 0:
                        m = markets[0]
                        tokens = json.loads(m.get("clobTokenIds", "[]"))
                        up_token = tokens[0] if tokens else ""
                        down_token = tokens[1] if len(tokens) > 1 else ""
                        if up_token:
                            print(f"  UP token:  {up_token[:20]}...")
                            print(f"  DN token:  {down_token[:20]}...")
                except Exception as e:
                    print(f"  Token fetch error: {e}")

            # Get LIVE prices from CLOB
            up_mid = up_buy = up_sell = spread_val = 0
            volume = 0

            if up_token:
                try:
                    # Midpoint — best estimate of true probability
                    mid_resp = clob.get_midpoint(up_token)
                    up_mid = float(mid_resp.get("mid", 0)) if isinstance(mid_resp, dict) else float(mid_resp)
                except Exception:
                    up_mid = 0

                try:
                    # Best executable BUY price
                    buy_resp = clob.get_price(up_token, "BUY")
                    up_buy = float(buy_resp.get("price", 0)) if isinstance(buy_resp, dict) else float(buy_resp)
                except Exception:
                    up_buy = 0

                try:
                    # Best executable SELL price
                    sell_resp = clob.get_price(up_token, "SELL")
                    up_sell = float(sell_resp.get("price", 0)) if isinstance(sell_resp, dict) else float(sell_resp)
                except Exception:
                    up_sell = 0

                try:
                    # Spread
                    spread_resp = clob.get_spread(up_token)
                    spread_val = float(spread_resp.get("spread", 0)) if isinstance(spread_resp, dict) else float(spread_resp)
                except Exception:
                    spread_val = up_sell - up_buy if (up_sell > 0 and up_buy > 0) else 0

            # Derive down prices
            down_mid = 1 - up_mid if up_mid > 0 else 0

            # Get volume from Gamma (cached per window)
            try:
                r = requests.get(f"{GAMMA_API}/markets",
                                 params={"slug": slug}, timeout=3)
                markets = r.json()
                if markets and isinstance(markets, list):
                    volume = float(markets[0].get("volume", 0))
            except Exception:
                pass

            # Store
            conn.execute("""
                INSERT INTO collections
                (timestamp, unix_ts, btc_price, market_id, market_question,
                 market_end_ts, up_price, down_price, up_token, down_token,
                 seconds_remaining, up_mid, up_buy, up_sell, spread)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now_dt.isoformat(), now, btc,
                slug, f"BTC Up or Down 5m @{window_ts}",
                window_ts + 300,
                up_mid, down_mid, up_token, down_token, secs_remaining,
                up_mid, up_buy, up_sell, spread_val
            ))
            conn.commit()

            # Display
            mid_str = f"{up_mid:>5.1%}" if up_mid > 0 else f"{'—':>5s}"
            buy_str = f"{up_buy:>5.1%}" if up_buy > 0 else f"{'—':>5s}"
            sell_str = f"{up_sell:>5.1%}" if up_sell > 0 else f"{'—':>5s}"
            sprd_str = f"{spread_val:>5.3f}" if spread_val > 0 else f"{'—':>5s}"

            print(f"  {now_dt.strftime('%H:%M:%S'):<10s} {slug[-6:]:>8s} "
                  f"{secs_remaining:>4d}s ${btc:>9,.2f} "
                  f"{mid_str} {buy_str} {sell_str} {sprd_str} "
                  f"${volume:>7,.0f}")

        except KeyboardInterrupt:
            print(f"\n  Stopped by user.")
            break
        except Exception as e:
            print(f"  Error: {e}")

        time.sleep(interval)

    # Stats
    count = conn.execute("SELECT COUNT(*) FROM collections").fetchone()[0]
    windows = conn.execute(
        "SELECT COUNT(DISTINCT market_id) FROM collections").fetchone()[0]
    print(f"\n  Total samples: {count}  |  Windows: {windows}")
    conn.close()
    print(f"  Database: {DB_PATH}")


def cmd_backtest(args):
    """Backtest the 4-minute rule on collected data."""
    trigger_min = getattr(args, "trigger", 4.0)
    trigger_secs = trigger_min * 60  # Convert to seconds
    min_btc_move = getattr(args, "min_move", 0.05)  # Min BTC % move to trigger
    fee_pct = 0.02  # Polymarket 2% winner fee

    print(f"\n{'='*90}")
    print(f"  BTC MOMENTUM BACKTEST")
    print(f"  Trigger: {trigger_min}m ({trigger_secs:.0f}s remaining)")
    print(f"  Min BTC move: {min_btc_move}%  |  Fee: {fee_pct:.0%}")
    print(f"{'='*90}")

    if not os.path.exists(DB_PATH):
        print(f"\n  ❌ No data. Run 'collect' first.")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT * FROM collections
        ORDER BY unix_ts ASC
    """).fetchall()

    if len(rows) < 10:
        print(f"\n  Only {len(rows)} samples. Need more data for meaningful backtest.")
        print(f"  Run: python -m engines.btc_momentum collect --minutes 120")
        return

    # Check data quality — are CLOB prices present?
    def safe_get(row, key, default=0):
        try:
            val = row[key]
            return val if val is not None else default
        except (KeyError, IndexError):
            return default

    clob_count = sum(1 for r in rows if safe_get(r, "up_mid", 0) > 0)
    stale_count = len(rows) - clob_count
    print(f"\n  {len(rows)} samples loaded")
    print(f"  Time range: {rows[0]['timestamp'][:19]} → {rows[-1]['timestamp'][:19]}")
    print(f"  CLOB prices: {clob_count} ({clob_count/len(rows)*100:.0f}%)")
    if stale_count > clob_count:
        print(f"  ⚠ Majority of data uses stale Gamma prices — results unreliable!")
        print(f"    Rerun collector: python -m engines.btc_momentum collect --minutes 360")

    # Group into 5-minute windows by market_id
    windows = {}
    for row in rows:
        mid = row["market_id"]
        if mid not in windows:
            windows[mid] = []
        windows[mid].append(dict(row))

    print(f"  {len(windows)} 5-minute windows")

    def run_backtest(t_secs, move_thresh, verbose=False):
        """Run backtest with specific trigger time and move threshold."""
        bt = {"win": 0, "loss": 0, "skip": 0, "pnl": 0,
              "up_win": 0, "up_loss": 0, "dn_win": 0, "dn_loss": 0}
        trade_list = []

        for window_start in sorted(windows.keys()):
            samples = windows[window_start]
            if len(samples) < 3:
                continue

            pre = [s for s in samples if s["seconds_remaining"] > t_secs]
            post = [s for s in samples if s["seconds_remaining"] <= t_secs]
            if not pre or not post:
                bt["skip"] += 1
                continue

            first_btc = pre[0]["btc_price"]
            trig_btc = pre[-1]["btc_price"]
            if first_btc == 0 or trig_btc == 0:
                bt["skip"] += 1
                continue

            move = (trig_btc - first_btc) / first_btc * 100
            if abs(move) < move_thresh:
                bt["skip"] += 1
                continue

            direction = "UP" if move > 0 else "DOWN"
            final_btc = post[-1]["btc_price"]
            final_move = (final_btc - first_btc) / first_btc * 100

            # Get entry price from CLOB
            if direction == "UP":
                entry = (post[0].get("up_buy") or post[0].get("up_mid")
                         or post[0].get("up_price") or 0.5)
            else:
                up_val = (post[0].get("up_sell") or post[0].get("up_mid")
                          or post[0].get("up_price") or 0.5)
                entry = max(1 - up_val, 0.01)

            if entry <= 0 or entry >= 1:
                entry = 0.5

            # Did momentum continue?
            won = (final_btc >= trig_btc) if direction == "UP" else (final_btc <= trig_btc)

            # P&L with fee (2% on winning payout)
            if won:
                gross = 1 - entry
                pnl = gross - fee_pct  # Win: payout $1, minus entry, minus fee
            else:
                pnl = -entry  # Lose: lose your entry

            bt["pnl"] += pnl
            if won:
                bt["win"] += 1
                if direction == "UP":
                    bt["up_win"] += 1
                else:
                    bt["dn_win"] += 1
            else:
                bt["loss"] += 1
                if direction == "UP":
                    bt["up_loss"] += 1
                else:
                    bt["dn_loss"] += 1

            trade_list.append({
                "window": window_start,
                "direction": direction,
                "btc_move_pct": move,
                "final_move_pct": final_move,
                "won": won,
                "entry_price": entry,
                "pnl": pnl,
            })

        return bt, trade_list

    # ── PRIMARY BACKTEST ──
    bt, trades = run_backtest(trigger_secs, min_btc_move, verbose=True)
    total = bt["win"] + bt["loss"]

    if total > 0:
        win_rate = bt["win"] / total
        avg_pnl = bt["pnl"] / total
        ev = avg_pnl  # Expected value per trade

        print(f"\n  ── PRIMARY RESULTS ──")
        print(f"  {'─'*55}")
        print(f"  Trigger:        {trigger_min}m  |  Min move: {min_btc_move}%")
        print(f"  Trades:         {total} ({bt['skip']} skipped)")
        print(f"  Win rate:       {bt['win']}/{total} ({win_rate:.1%})")
        print(f"  Total P&L:      ${bt['pnl']:+.2f} (after {fee_pct:.0%} fee)")
        print(f"  Avg P&L/trade:  ${avg_pnl:+.4f}")
        print(f"  Expected value: ${ev:+.4f}/trade")

        # Direction breakdown
        up_total = bt["up_win"] + bt["up_loss"]
        dn_total = bt["dn_win"] + bt["dn_loss"]
        print(f"\n  Direction Breakdown:")
        if up_total > 0:
            print(f"    UP trades:   {up_total:>4d}  |  Win rate: {bt['up_win']/up_total:.1%}")
        if dn_total > 0:
            print(f"    DOWN trades: {dn_total:>4d}  |  Win rate: {bt['dn_win']/dn_total:.1%}")

        # Show recent trades
        if trades:
            print(f"\n  Last 15 Trades:")
            print(f"  {'Dir':>4s} {'BTC%':>7s} {'Final%':>7s} {'Entry':>6s} "
                  f"{'P&L':>7s} {'Result'}")
            print(f"  {'─'*50}")
            for t in trades[-15:]:
                result = "✅ WIN" if t["won"] else "❌ LOSS"
                print(f"  {t['direction']:>4s} {t['btc_move_pct']:>+6.3f}% "
                      f"{t['final_move_pct']:>+6.3f}% {t['entry_price']:>5.1%} "
                      f"${t['pnl']:>+6.3f} {result}")

    else:
        print(f"\n  No trades generated. Need more data with price variation.")
        print(f"  Skipped: {bt['skip']} windows")
        conn.close()
        return

    # ── TRIGGER TIME OPTIMIZATION ──
    if total >= 5:
        print(f"\n  ── TRIGGER TIME SWEEP (min move: {min_btc_move}%) ──")
        print(f"  {'Trigger':>8s} {'Trades':>7s} {'WinRate':>8s} "
              f"{'AvgP&L':>8s} {'TotalP&L':>10s} {'EV':>8s}")
        print(f"  {'─'*55}")

        for t_test in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 3.8, 4.0, 4.2, 4.5]:
            t_bt, _ = run_backtest(t_test * 60, min_btc_move)
            t_total = t_bt["win"] + t_bt["loss"]
            if t_total > 0:
                wr = t_bt["win"] / t_total
                avg = t_bt["pnl"] / t_total
                flag = " ✅" if avg > 0 else ""
                print(f"  {t_test:>6.1f}m {t_total:>7d} {wr:>7.1%} "
                      f"${avg:>7.4f} ${t_bt['pnl']:>9.2f} ${avg:>7.4f}{flag}")

    # ── BTC MOVE THRESHOLD SWEEP ──
    if total >= 5:
        print(f"\n  ── BTC MOVE THRESHOLD SWEEP (trigger: {trigger_min}m) ──")
        print(f"  {'MinMove':>8s} {'Trades':>7s} {'WinRate':>8s} "
              f"{'AvgP&L':>8s} {'TotalP&L':>10s} {'EV':>8s}")
        print(f"  {'─'*55}")

        for m_test in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]:
            m_bt, _ = run_backtest(trigger_secs, m_test)
            m_total = m_bt["win"] + m_bt["loss"]
            if m_total > 0:
                wr = m_bt["win"] / m_total
                avg = m_bt["pnl"] / m_total
                flag = " ✅" if avg > 0 else ""
                print(f"  {m_test:>6.2f}% {m_total:>7d} {wr:>7.1%} "
                      f"${avg:>7.4f} ${m_bt['pnl']:>9.2f} ${avg:>7.4f}{flag}")

    conn.close()
    print(f"\n{'='*90}")


def cmd_stats(args):
    """Show database statistics."""
    if not os.path.exists(DB_PATH):
        print(f"  No database found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    count = conn.execute("SELECT COUNT(*) FROM collections").fetchone()[0]
    if count == 0:
        print(f"  Database is empty.")
        return

    first = conn.execute("SELECT MIN(timestamp) FROM collections").fetchone()[0]
    last = conn.execute("SELECT MAX(timestamp) FROM collections").fetchone()[0]
    windows = conn.execute(
        "SELECT COUNT(DISTINCT unix_ts / 300) FROM collections").fetchone()[0]
    avg_btc = conn.execute("SELECT AVG(btc_price) FROM collections").fetchone()[0]

    print(f"\n  BTC Momentum Database Stats:")
    print(f"  {'─'*40}")
    print(f"  Samples:      {count:,}")
    print(f"  5-min windows: {windows:,}")
    print(f"  First:        {first[:19]}")
    print(f"  Last:         {last[:19]}")
    print(f"  Avg BTC:      ${avg_btc:,.2f}")
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="BTC 5-Minute Momentum Bot")
    subs = parser.add_subparsers(dest="command", required=True)

    subs.add_parser("discover", help="Find active 5-min BTC markets")

    p_collect = subs.add_parser("collect", help="Collect price data")
    p_collect.add_argument("--minutes", type=int, default=30)
    p_collect.add_argument("--interval", type=int, default=10)

    p_bt = subs.add_parser("backtest", help="Backtest trigger thresholds")
    p_bt.add_argument("--trigger", type=float, default=4.0,
                       help="Trigger time in minutes (default: 4.0)")
    p_bt.add_argument("--min-move", type=float, default=0.05,
                       help="Min BTC move %% to trigger (default: 0.05)")

    subs.add_parser("stats", help="Show database stats")

    args = parser.parse_args()
    dispatch = {
        "discover": lambda a: discover_5min_markets(),
        "collect": cmd_collect,
        "backtest": cmd_backtest,
        "stats": cmd_stats,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
