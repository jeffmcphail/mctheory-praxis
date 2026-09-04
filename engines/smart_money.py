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
import contextlib
import json
import os
import signal
import sqlite3
import sys
import time
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv
load_dotenv()

# Cycle 64: this module prints box-drawing characters and emoji throughout. Under
# a cp1252 stdout (any invocation that does not set PYTHONUTF8=1 -- the service
# bat does, an ad-hoc run does not) those raise UnicodeEncodeError and kill the
# process mid-run. That matters beyond cosmetics: an unhandled exception exits 1,
# which is also EXIT_INCOMPLETE, so a display failure could forge or mask the
# run's real status. errors="replace" guarantees output can never crash the
# collector, whatever the console can represent.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

DATA_API = "https://data-api.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "smart_money.db"  # Cycle 47 (44h-bulk): anchor to repo root

# Tracker config
DEFAULT_TOP_N = 25           # Track top N traders
MIN_POSITION_USD = 100       # Ignore positions under $100
SNAPSHOT_INTERVAL = 300      # Snapshot every 5 minutes
CONVERGENCE_THRESHOLD = 3    # N traders in same market = strong signal

CATEGORIES = ["OVERALL", "POLITICS", "SPORTS", "CRYPTO", "CULTURE"]

# ── Cycle 64: crash-safety / gap-detection constants ──────────────────────
# Cycle 63 T4 found a run that stopped at wallet 416/1571 having committed
# nothing, because the whole 1,586-wallet loop sat inside ONE transaction that
# committed only at the end. Cycle 64 T1 then established the cause was a HOST
# unexpected shutdown (System event 6008: "previous system shutdown at 4:25:57
# AM on 9/3/2026 was unexpected"), not a fault in this module -- every collector
# on the box stopped inside the same four minutes. A power loss never runs a
# `finally` block, so a context manager alone would not have saved those rows.
# Incremental commit would have: wallets 1-415 were lost only because nothing
# had been written yet.
COLLECTOR_NAME = "smart_money"
COLLECTOR_VENUE = "polymarket"

DEFAULT_BATCH_SIZE = 50      # wallets per commit; bounds loss to <= this many
DEFAULT_CADENCE_HOURS = 6.0  # PraxisSmartMoney fires every 6h
DEFAULT_STALENESS_MARGIN_HOURS = 2.0

# Honest exit codes (memory #12 -- exit-code honesty). Task Scheduler records
# LastTaskResult, so an incomplete run must be distinguishable from a complete
# one WITHOUT reading the log.
EXIT_OK = 0
EXIT_INCOMPLETE = 1   # ran, but did not cover every wallet
EXIT_STALE = 2        # most recent snapshot older than cadence + margin
EXIT_FATAL = 3        # could not run at all (DB error, no wallets)

# collector_gaps lives in smart_money.db, NOT crypto_data.db -- the Cycle 64
# brief holds the 24 GiB main DB read-only. Column-for-column identical to the
# Cycle 62A table so the two can be consolidated later without a migration.
COLLECTOR_GAPS_DDL = """
    CREATE TABLE IF NOT EXISTS collector_gaps (
        collector   TEXT    NOT NULL,
        venue       TEXT    NOT NULL,
        timestamp   INTEGER NOT NULL,
        datetime    TEXT    NOT NULL,
        gap_end     INTEGER,
        gap_seconds REAL,
        reason      TEXT,
        detected_at INTEGER NOT NULL,
        PRIMARY KEY (collector, venue, timestamp)
    )
"""

# smart_money has NO backfill path: the Polymarket data-api serves CURRENT
# positions only, so position_snapshots can only ever be built by sampling
# forward. A missed snapshot is gone permanently. Gaps are therefore recorded
# CLOSED with this prefix rather than left open, so no retry logic ever treats
# them as pending work.
UNFILLABLE = "UNFILLABLE"


# ═══════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════

# Cycle 64: set by main() from --db. Lets the deliberate-kill verification run
# against a scratch database instead of writing partial snapshots into the
# production table. None means "use DB_PATH".
_DB_OVERRIDE = None


def active_db_path():
    return Path(_DB_OVERRIDE) if _DB_OVERRIDE else DB_PATH


def init_db():
    path = active_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    # Cycle 64: the nightly mirror and this collector can touch the file at the
    # same moment; wait rather than raising "database is locked" instantly.
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute(COLLECTOR_GAPS_DDL)

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

    # Cycle 64 FIX: this DDL had drifted from the live table. It still described
    # the pre-Cycle-25 shape -- an AUTOINCREMENT id, `timestamp` as TEXT, and no
    # `datetime` column at all -- while _insert_position_row writes the migrated
    # Rule 35 shape (ms `timestamp` + ISO `datetime`, compound PK). On the
    # existing database CREATE TABLE IF NOT EXISTS is a no-op, so the drift was
    # invisible. On a FRESH machine it would not be: init_db() would create the
    # stale schema and every insert would fail with "table position_snapshots
    # has no column named datetime". That is precisely the MCRMINI2 case, so it
    # is fixed here to match the live schema exactly.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS position_snapshots (
            snapshot_id TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            datetime TEXT NOT NULL,
            wallet TEXT NOT NULL,
            market_slug TEXT,
            market_title TEXT,
            outcome TEXT,
            size REAL,
            avg_price REAL,
            current_price REAL,
            value_usd REAL,
            pnl_usd REAL,
            PRIMARY KEY (snapshot_id, wallet, market_slug, outcome)
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


# ── Cycle 64: gap recording, staleness, alerting ──────────────────────────

def _now_ms():
    return int(time.time() * 1000)


def _ms_to_iso(ms):
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat()


def record_gap(conn, start_ms, end_ms, reason, verbose=True):
    """Persist a collection gap, mirroring engines/collector_common.record_gap.

    Idempotent on (collector, venue, timestamp). Unlike the Cycle 62A
    collectors these gaps are written CLOSED (gap_end set) and reason-prefixed
    UNFILLABLE, because smart_money cannot be backfilled -- see the note on the
    UNFILLABLE constant. Leaving them open would invite a retry that can never
    succeed.
    """
    gap_seconds = None if end_ms is None else (end_ms - start_ms) / 1000.0
    conn.execute(
        """
        INSERT INTO collector_gaps
            (collector, venue, timestamp, datetime, gap_end, gap_seconds,
             reason, detected_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(collector, venue, timestamp) DO UPDATE SET
            gap_end     = excluded.gap_end,
            gap_seconds = excluded.gap_seconds,
            reason      = excluded.reason
        """,
        (COLLECTOR_NAME, COLLECTOR_VENUE, start_ms, _ms_to_iso(start_ms),
         end_ms, gap_seconds, reason, _now_ms()),
    )
    conn.commit()
    if verbose:
        span = "" if gap_seconds is None else f" ({gap_seconds/3600:.2f}h)"
        print(f"  GAP RECORDED  {COLLECTOR_NAME}/{COLLECTOR_VENUE} "
              f"{_ms_to_iso(start_ms)}{span}")
        print(f"                reason: {reason}")


def latest_snapshot_dt(conn):
    """UTC datetime of the most recent committed snapshot, or None."""
    row = conn.execute(
        "SELECT MAX(snapshot_id) FROM position_snapshots").fetchone()
    if not row or not row[0]:
        return None
    try:
        return datetime.strptime(row[0], "%Y%m%d_%H%M%S").replace(
            tzinfo=timezone.utc)
    except ValueError:
        return None


def check_staleness(conn, cadence_hours, margin_hours, verbose=True):
    """Is the most recent snapshot older than cadence + margin?

    Returns (is_stale, age_hours, latest_dt). Cycle 63 T4's 34.7h outage went
    unremarked for over a day because nothing ever asked this question.
    """
    latest = latest_snapshot_dt(conn)
    if latest is None:
        if verbose:
            print("  STALENESS: no snapshots on record")
        return True, None, None
    age_h = (datetime.now(timezone.utc) - latest).total_seconds() / 3600.0
    limit = cadence_hours + margin_hours
    stale = age_h > limit
    if verbose:
        flag = "STALE" if stale else "ok"
        print(f"  STALENESS: latest={latest:%Y-%m-%d %H:%M:%S}Z "
              f"age={age_h:.2f}h limit={limit:.2f}h -> {flag}")
    return stale, age_h, latest


def resolve_alert_url():
    """PRAXIS_ALERT_URL, falling back to legacy TEAMS_WEBHOOK_URL.

    Same resolution order as scripts/funding_regime_alert.py.
    """
    url = os.getenv("PRAXIS_ALERT_URL", "").strip()
    if url:
        return url, "PRAXIS_ALERT_URL"
    url = os.getenv("TEAMS_WEBHOOK_URL", "").strip()
    if url:
        return url, "TEAMS_WEBHOOK_URL (legacy fallback)"
    return "", ""


def post_alert(url, body, title):
    """POST an ASCII push to the ntfy.sh-style backend. Returns (ok, msg)."""
    req = urllib.request.Request(
        url, data=body.encode("utf-8"),
        headers={
            "Content-Type": "text/plain; charset=utf-8",
            "Title": title,
            "Tags": "rotating_light",
            "Priority": "4",
            "Markdown": "yes",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            txt = resp.read().decode("utf-8", errors="replace")
            return resp.status < 300, f"HTTP {resp.status}: {txt[:160]}"
    except Exception as e:
        return False, f"ERROR: {type(e).__name__}: {e}"[:200]


def fire_alert(body, title, do_alert, verbose=True):
    """Send an alert if --alert was passed. Returns True if a POST succeeded."""
    if not do_alert:
        if verbose:
            print(f"  ALERT (dry-run, --alert not passed): {title}")
        return False
    url, src = resolve_alert_url()
    if not url:
        print("  ALERT: no PRAXIS_ALERT_URL / TEAMS_WEBHOOK_URL set; "
              "cannot notify")
        return False
    ok, msg = post_alert(url, body, title)
    print(f"  ALERT via {src}: {'sent' if ok else 'FAILED'} -- {msg}")
    return ok


def _insert_position_row(conn, snapshot_id, now_iso, now_ms, address,
                          slug, title, outcome, size, avg_price,
                          cur_price, value, pnl):
    """Single-write into the post-Cycle-25-cutover position_snapshots
    table (Rule 35: ms timestamp + datetime + compound PK on
    (snapshot_id, wallet, market_slug, outcome)). INSERT OR REPLACE
    preserves the existing UPSERT semantics on the natural key.
    """
    conn.execute("""
        INSERT OR REPLACE INTO position_snapshots
        (snapshot_id, timestamp, datetime, wallet, market_slug,
         market_title, outcome, size, avg_price, current_price,
         value_usd, pnl_usd)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (snapshot_id, now_ms, now_iso, address, slug,
          str(title)[:100], outcome, size, avg_price, cur_price,
          value, pnl))


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


def fetch_positions(wallet_address, raise_on_error=False):
    """Fetch current positions for a wallet.

    Cycle 64: `raise_on_error` exists because the default swallow-and-return-[]
    makes a failed fetch indistinguishable from a wallet that genuinely holds
    no positions. The snapshot loop needs that distinction to report the failed
    set (T2), so it passes True; every other caller keeps the old behaviour.
    """
    try:
        r = requests.get(f"{DATA_API}/positions", params={
            "user": wallet_address,
        }, timeout=15)

        if r.status_code == 200:
            return r.json()
        if raise_on_error:
            raise RuntimeError(f"HTTP {r.status_code}")
        return []
    except Exception as e:
        if raise_on_error:
            raise
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
    """Take a position snapshot for all tracked wallets. Returns an exit code.

    Cycle 64 T2. Three properties the pre-Cycle-64 version lacked:

      1. The connection closes on EVERY exit path (contextlib.closing), so an
         unhandled exception no longer leaves the WAL sidecars orphaned.
      2. Rows commit every --batch-size wallets instead of once at the end. The
         2026-09-03 host shutdown cost all 1,586 wallets because nothing had
         been written when the power went; with batching the loss is bounded to
         at most one batch. Note that this, not the context manager, is what
         actually survives a power loss -- no `finally` runs when the machine
         dies.
      3. The process exits non-zero when the run is incomplete, so Task
         Scheduler's LastTaskResult distinguishes "1,586 of 1,586" from
         "416 of 1,586" without anyone reading the log.
    """
    batch_size = max(1, int(getattr(args, "batch_size", DEFAULT_BATCH_SIZE)))
    verbose = int(getattr(args, "verbose", 2))
    do_alert = bool(getattr(args, "alert", False))
    wallet_limit = getattr(args, "limit", None)
    cadence_h = float(getattr(args, "cadence_hours", DEFAULT_CADENCE_HOURS))
    margin_h = float(getattr(args, "staleness_margin_hours",
                             DEFAULT_STALENESS_MARGIN_HOURS))

    # Cooperative stop flag. On Windows only SIGINT is reliably deliverable to
    # a Python handler -- `taskkill /F` and a power loss terminate the process
    # outright and no handler runs. That is deliberate in the design: the
    # incremental commit below is what protects the data; these handlers only
    # make a *polite* stop tidy.
    state = {"stop": False, "reason": ""}

    def _request_stop(signum, _frame):
        state["stop"] = True
        state["reason"] = f"signal {signum}"
        print(f"\n  STOP REQUESTED (signal {signum}) -- "
              f"committing current batch and closing cleanly")

    for _sig in ("SIGINT", "SIGTERM", "SIGBREAK"):
        s = getattr(signal, _sig, None)
        if s is not None:
            try:
                signal.signal(s, _request_stop)
            except (ValueError, OSError):
                pass  # not on the main thread, or unsupported on this platform

    with contextlib.closing(init_db()) as conn:
        stale, age_h, latest = check_staleness(conn, cadence_h, margin_h,
                                               verbose=verbose >= 1)

        wallets = conn.execute(
            "SELECT address, username FROM tracked_wallets WHERE active=1"
        ).fetchall()
        if wallet_limit:
            wallets = wallets[:int(wallet_limit)]

        if not wallets:
            print("  No tracked wallets. Run: "
                  "python -m engines.smart_money discover")
            return EXIT_FATAL

        snapshot_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        now_iso = datetime.now(timezone.utc).isoformat()
        now_ms = int(time.time() * 1000)
        started_ms = now_ms
        total = len(wallets)

        print(f"\n{'='*90}")
        print(f"  POSITION SNAPSHOT - {snapshot_id}")
        print(f"  Tracking {total} wallets  (batch_size={batch_size}, "
              f"commit every {batch_size} wallets)")
        print(f"{'='*90}")

        total_positions = 0
        completed = 0
        committed_through = 0
        failed = []
        last_progress = time.time()

        for i, (address, username) in enumerate(wallets):
            if state["stop"]:
                break

            try:
                positions = fetch_positions(address, raise_on_error=True)
            except Exception as e:
                # T2: a per-wallet failure must not kill the run.
                failed.append((address, f"{type(e).__name__}: {e}"[:120]))
                if verbose >= 3:
                    print(f"    WALLET FAILED {address[:12]}...: {e}")
                time.sleep(0.3)
                continue

            for p in (positions or []):
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

                _insert_position_row(
                    conn, snapshot_id, now_iso, now_ms,
                    address, slug, title, outcome, size, avg_price,
                    cur_price, value, pnl,
                )
                total_positions += 1

            completed = i + 1

            # T2: incremental commit. Bounds worst-case loss to one batch.
            if completed % batch_size == 0:
                conn.commit()
                committed_through = completed
                if verbose >= 3:
                    print(f"    COMMIT through wallet {completed}/{total}")

            time.sleep(0.3)  # Rate limit

            now_t = time.time()
            if now_t - last_progress >= 15 or completed == total:
                print(f"    [{datetime.now().strftime('%H:%M:%S')}] "
                      f"{completed}/{total} wallets | "
                      f"{total_positions} positions captured")
                last_progress = now_t

        conn.commit()
        committed_through = completed

        incomplete = completed < total
        print(f"\n  Snapshot {snapshot_id}: {total_positions} positions from "
              f"{completed}/{total} wallets "
              f"({'INCOMPLETE' if incomplete else 'complete'})")
        if failed:
            print(f"  Wallets failed (fetch error, skipped): {len(failed)}")
            for addr, err in failed[:10]:
                print(f"    {addr[:16]}...  {err}")
            if len(failed) > 10:
                print(f"    ... and {len(failed)-10} more")

        # T3: an incomplete run leaves a durable, terminal trace.
        exit_code = EXIT_OK
        if incomplete:
            exit_code = EXIT_INCOMPLETE
            reason = (f"{UNFILLABLE}: incomplete run {snapshot_id} -- "
                      f"{completed}/{total} wallets"
                      + (f" (stopped: {state['reason']})" if state["reason"] else ""))
            record_gap(conn, started_ms, _now_ms(), reason,
                       verbose=verbose >= 1)
            fire_alert(
                f"**smart_money snapshot INCOMPLETE**\n"
                f"snapshot_id: {snapshot_id}\n"
                f"covered: {completed}/{total} wallets\n"
                f"positions kept: {total_positions}\n"
                f"failed fetches: {len(failed)}\n"
                f"This gap is UNFILLABLE -- Polymarket serves current "
                f"positions only.",
                "SMART MONEY SNAPSHOT INCOMPLETE", do_alert,
                verbose=verbose >= 1)
        elif stale:
            # The run completed, but it arrived late enough that at least one
            # scheduled snapshot was missed. Record + notify, then report it.
            exit_code = EXIT_STALE
            reason = (f"{UNFILLABLE}: staleness -- previous snapshot "
                      f"{latest:%Y-%m-%d %H:%M:%S}Z was {age_h:.2f}h old "
                      f"(cadence {cadence_h}h + margin {margin_h}h)"
                      if latest else f"{UNFILLABLE}: no prior snapshot")
            if latest is not None:
                record_gap(conn, int(latest.timestamp()*1000), started_ms,
                           reason, verbose=verbose >= 1)
            fire_alert(
                f"**smart_money snapshots were STALE**\n"
                f"previous: {latest:%Y-%m-%d %H:%M:%S}Z ({age_h:.2f}h ago)\n"
                f"cadence {cadence_h}h + margin {margin_h}h exceeded\n"
                f"This run ({snapshot_id}) completed {completed}/{total}."
                if latest else "smart_money has no prior snapshot on record.",
                "SMART MONEY SNAPSHOTS STALE", do_alert, verbose=verbose >= 1)

        # Display top holdings (informational; never affects the exit code).
        try:
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
        except Exception as e:
            top_markets = []
            print(f"  (top-holdings rollup skipped: {e})")

        # ASCII-only, and exception-proof. Cycle 64 verification caught this the
        # hard way: the box-drawing separator raised UnicodeEncodeError under a
        # cp1252 stdout, and an unhandled exception exits 1 -- the same code as
        # EXIT_INCOMPLETE. A cosmetic rollup must never be able to forge or mask
        # the run's real exit status.
        if top_markets and verbose >= 2:
            try:
                print("\n  Markets with multiple top traders:")
                print(f"  {'Market':<45s} {'Side':<5s} {'Wallets':>8s} "
                      f"{'Total$':>10s} {'Price':>6s}")
                print(f"  {'-'*80}")
                for m in top_markets:
                    flag = "**" if m[2] >= CONVERGENCE_THRESHOLD else "  "
                    print(f"  {flag}{str(m[0])[:43]:<45s} {m[1]:<5s} {m[2]:>8d} "
                          f"${m[3]:>9,.0f} {m[4]:>5.0%}")
            except Exception as e:
                print(f"  (top-holdings display skipped: "
                      f"{type(e).__name__}: {e})")

        print(f"\n  committed through wallet: {committed_through}/{total}")
        print(f"  exit code: {exit_code} "
              f"({'OK' if exit_code == EXIT_OK else 'INCOMPLETE' if exit_code == EXIT_INCOMPLETE else 'STALE'})")
        print(f"\n{'='*90}")
        return exit_code


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
    if not active_db_path().exists():
        print("  No data. Run discover + snapshot + diff first.")
        return

    conn = sqlite3.connect(str(active_db_path()))
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
            now_iso = datetime.now(timezone.utc).isoformat()
            now_ms = int(time.time() * 1000)
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

                    _insert_position_row(
                        conn, snapshot_id, now_iso, now_ms,
                        address, slug, title, outcome, size, avg_price,
                        cur_price, value, (cur_price - avg_price) * size,
                    )
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
    parser.add_argument("--db", default=None,
                        help="Override the database path (default: "
                             "data/smart_money.db). Used by the Cycle 64 "
                             "deliberate-kill verification so test runs do not "
                             "write partial snapshots into production.")
    subs = parser.add_subparsers(dest="command")

    p_disc = subs.add_parser("discover", help="Find top traders")
    p_disc.add_argument("--category", default="OVERALL",
                        help="OVERALL, POLITICS, SPORTS, CRYPTO, CULTURE, or ALL")
    p_disc.add_argument("--top", type=int, default=DEFAULT_TOP_N)
    p_disc.add_argument("--period", default="MONTH",
                        help="MONTH, WEEK, ALL")

    p_snap = subs.add_parser("snapshot", help="Snapshot all tracked positions")
    p_snap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Wallets per commit (default {DEFAULT_BATCH_SIZE}). "
                             "Bounds worst-case row loss on a host crash.")
    p_snap.add_argument("--cadence-hours", type=float,
                        default=DEFAULT_CADENCE_HOURS,
                        help=f"Expected schedule cadence (default "
                             f"{DEFAULT_CADENCE_HOURS}h)")
    p_snap.add_argument("--staleness-margin-hours", type=float,
                        default=DEFAULT_STALENESS_MARGIN_HOURS,
                        help="Grace beyond the cadence before a run counts as "
                             f"stale (default {DEFAULT_STALENESS_MARGIN_HOURS}h)")
    p_snap.add_argument("--alert", action="store_true",
                        help="POST to PRAXIS_ALERT_URL on incomplete/stale. "
                             "Without it the alert is computed but not sent.")
    p_snap.add_argument("--limit", type=int, default=None,
                        help="Only process the first N wallets (test aid)")
    p_snap.add_argument("--verbose", type=int, default=3,
                        help="0=quiet 1=gaps 2=normal 3=max (default 3)")
    p_snap.add_argument("--validate", dest="validate",
                        action="store_true", default=True,
                        help="Re-read the DB after the run and verify the "
                             "committed row count (default on)")
    p_snap.add_argument("--no-validate", dest="validate", action="store_false",
                        help="Skip the post-run read-back")

    p_gaps = subs.add_parser("gaps", help="Show recorded collector_gaps rows")
    p_gaps.add_argument("--limit", type=int, default=20)

    p_stale = subs.add_parser("staleness",
                              help="Check snapshot freshness; non-zero if stale")
    p_stale.add_argument("--cadence-hours", type=float,
                         default=DEFAULT_CADENCE_HOURS)
    p_stale.add_argument("--staleness-margin-hours", type=float,
                         default=DEFAULT_STALENESS_MARGIN_HOURS)
    p_stale.add_argument("--alert", action="store_true")

    subs.add_parser("diff", help="Compare recent snapshots")
    subs.add_parser("signals", help="Show convergence signals")

    p_mon = subs.add_parser("monitor", help="Continuous monitoring")
    p_mon.add_argument("--interval", type=int, default=SNAPSHOT_INTERVAL)

    p_prof = subs.add_parser("profile", help="Deep dive on a wallet")
    p_prof.add_argument("address", type=str)

    args = parser.parse_args()

    global _DB_OVERRIDE
    if getattr(args, "db", None):
        _DB_OVERRIDE = args.db
        print(f"  DB OVERRIDE: {_DB_OVERRIDE}")

    if args.command == "discover":
        cmd_discover(args)
    elif args.command == "snapshot":
        code = cmd_snapshot(args)
        if getattr(args, "validate", False):
            code = _validate_after_run(code, args)
        return code
    elif args.command == "gaps":
        return cmd_gaps(args)
    elif args.command == "staleness":
        return cmd_staleness(args)
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
    return EXIT_OK


def _validate_after_run(code, args):
    """Read the DB back after the run and report what actually landed.

    Cycle 64 verification discipline: the brief requires proof by measurement,
    so the collector states what it committed rather than what it believes it
    committed. Never downgrades a failing exit code to success.
    """
    with contextlib.closing(init_db()) as conn:
        latest = latest_snapshot_dt(conn)
        row = conn.execute(
            "SELECT snapshot_id, COUNT(*), COUNT(DISTINCT wallet) "
            "FROM position_snapshots "
            "WHERE snapshot_id = (SELECT MAX(snapshot_id) FROM position_snapshots)"
        ).fetchone()
        print("  VALIDATE (read-back):")
        if row and row[0]:
            print(f"    latest snapshot_id : {row[0]}")
            print(f"    rows committed     : {row[1]}")
            print(f"    distinct wallets   : {row[2]}")
        else:
            print("    no snapshot rows found")
            code = code or EXIT_FATAL
        n_gaps = conn.execute(
            "SELECT COUNT(*) FROM collector_gaps WHERE collector=?",
            (COLLECTOR_NAME,)).fetchone()[0]
        print(f"    collector_gaps rows: {n_gaps}")
    return code


def cmd_gaps(args):
    """List recorded gaps. All smart_money gaps are terminal by construction."""
    with contextlib.closing(init_db()) as conn:
        rows = conn.execute(
            "SELECT datetime, gap_end, gap_seconds, reason, detected_at "
            "FROM collector_gaps WHERE collector=? "
            "ORDER BY timestamp DESC LIMIT ?",
            (COLLECTOR_NAME, int(getattr(args, "limit", 20))),
        ).fetchall()
    print(f"\n  collector_gaps -- {COLLECTOR_NAME}/{COLLECTOR_VENUE} "
          f"({len(rows)} shown)")
    if not rows:
        print("    (none)")
        return EXIT_OK
    for dt, gend, gsec, reason, det in rows:
        span = f"{gsec/3600:.2f}h" if gsec is not None else "OPEN"
        print(f"    {dt}  span={span:>8s}  {reason}")
    return EXIT_OK


def cmd_staleness(args):
    """Standalone freshness probe. Non-zero + alert when stale."""
    with contextlib.closing(init_db()) as conn:
        stale, age_h, latest = check_staleness(
            conn, float(args.cadence_hours),
            float(args.staleness_margin_hours), verbose=True)
        if stale:
            fire_alert(
                (f"**smart_money snapshots are STALE**\n"
                 f"latest: {latest:%Y-%m-%d %H:%M:%S}Z ({age_h:.2f}h ago)\n"
                 f"cadence {args.cadence_hours}h + margin "
                 f"{args.staleness_margin_hours}h exceeded."
                 if latest else "smart_money has no snapshots on record."),
                "SMART MONEY SNAPSHOTS STALE",
                bool(getattr(args, "alert", False)))
            return EXIT_STALE
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
