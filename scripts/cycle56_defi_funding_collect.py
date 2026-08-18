"""
scripts/cycle56_defi_funding_collect.py
=======================================
Cycle 56 RECON — DeFi perp funding-rate collection (READ-ONLY public data).

Pulls historical funding from Hyperliquid (raw public /info endpoint) and
dYdX v4 (public Indexer) for a candidate carry universe, into an ISOLATED
research DB:  data/defi_funding_research.db

WHY A SEPARATE DB (not the production funding_rates):
  The live monitor (scripts/funding_monitor.py:169) reads funding_rates
  WITHOUT a venue filter, then drop_duplicates("timestamp"). CEX rows are
  8-hourly; Hyperliquid/dYdX are HOURLY. Writing hourly DeFi rows into the
  production table would inject ~24 rows/day that mostly don't collide with
  the 8h CEX timestamps, corrupting the monitor's funding features
  (annualization, positive-share) -> bad funding_signals/alerts. Until the
  monitor read is venue-scoped (a prerequisite for production integration),
  DeFi funding stays in this isolated research DB. Structural > procedural
  (Cycle 54 isolation lesson).

NO credentials, NO orders, NO wallet keys. Funding history is public on both
venues. Idempotent: INSERT OR IGNORE on (asset, venue, timestamp).

Schema mirrors production funding_rates. NOTE the cadence difference recorded
per venue: hyperliquid/dydx = HOURLY rate; binance/bybit (production) = 8H
rate. Any cross-venue analysis MUST annualize per-venue cadence.
"""
from __future__ import annotations

import json
import sqlite3
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESEARCH_DB = REPO / "data" / "defi_funding_research.db"

# Uniform window for clean cross-venue/cross-asset comparison. 2.5y of hourly
# data is ample for carry stats and covers the atlas Exp 13 OOS (2025-01-01+)
# and the 2025 "carry Sharpe went negative" regime (BIS/SSRN).
START_MS = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
NOW_MS = int(datetime.now(timezone.utc).timestamp() * 1000)
HOUR_MS = 3_600_000

# Hyperliquid candidate universe (coin names as Hyperliquid lists them).
HL_MAJORS = ["BTC", "ETH", "SOL", "XRP", "ADA", "AVAX"]
HL_ALTS = ["LINK", "DOGE", "LTC", "ARB", "AAVE", "UNI", "BNB", "APT",
           "SUI", "OP", "CRV", "MKR", "INJ", "ATOM"]   # broadly Canadian-hedgeable
HL_TAIL = ["HYPE", "kPEPE", "kSHIB", "WLD"]            # HL-native / meme tail
HL_ASSETS = HL_MAJORS + HL_ALTS + HL_TAIL

# dYdX v4 majors only (small/declining venue; majors are where depth is).
DYDX_TICKERS = {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD",
                "XRP": "XRP-USD", "ADA": "ADA-USD", "AVAX": "AVAX-USD"}

HL_INFO = "https://api.hyperliquid.xyz/info"
DYDX_INDEXER = "https://indexer.dydx.trade/v4/historicalFunding"


def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(RESEARCH_DB))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS funding_rates (
            asset        TEXT    NOT NULL,
            venue        TEXT    NOT NULL,
            timestamp    INTEGER NOT NULL,
            datetime     TEXT    NOT NULL,
            funding_rate REAL    NOT NULL,
            cadence      TEXT    NOT NULL,   -- 'hourly' for DeFi (vs 8h CEX)
            PRIMARY KEY (asset, venue, timestamp)
        )
    """)
    conn.commit()
    return conn


def _post_json(url: str, payload: dict, timeout: int = 25) -> object:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _get_json(url: str, timeout: int = 25) -> object:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S+00:00")


def _store(conn, rows):
    conn.executemany(
        "INSERT OR IGNORE INTO funding_rates "
        "(asset, venue, timestamp, datetime, funding_rate, cadence) "
        "VALUES (?, ?, ?, ?, ?, ?)", rows)
    conn.commit()


def collect_hyperliquid(conn):
    for asset in HL_ASSETS:
        stored = seen = 0
        start = START_MS
        pages = 0
        try:
            while start < NOW_MS and pages < 80:
                pages += 1
                for attempt in range(4):
                    try:
                        data = _post_json(HL_INFO, {
                            "type": "fundingHistory",
                            "coin": asset, "startTime": start, "endTime": NOW_MS})
                        break
                    except urllib.error.HTTPError as e:
                        if e.code == 429:
                            time.sleep(1.5 * (attempt + 1)); continue
                        raise
                else:
                    print(f"  HL {asset}: rate-limited out, partial"); break
                if not data:
                    break
                rows = []
                last_t = start
                for d in data:
                    t = int(d["time"])
                    last_t = max(last_t, t)
                    th = (t // HOUR_MS) * HOUR_MS
                    rows.append((asset, "hyperliquid", th, _iso(th),
                                 float(d["fundingRate"]), "hourly"))
                seen += len(data)
                _store(conn, rows)
                stored += conn.total_changes  # approximate; reset below
                conn.total_changes  # noqa
                if last_t <= start:
                    break
                start = last_t + 1
                time.sleep(0.15)
            n = conn.execute("SELECT COUNT(*) FROM funding_rates WHERE "
                             "venue='hyperliquid' AND asset=?", (asset,)).fetchone()[0]
            print(f"  HL {asset:<7} pages={pages:<3} api_records={seen:<6} db_rows={n}")
        except Exception as e:
            print(f"  HL {asset}: ERROR {e!r}")


def collect_dydx(conn):
    for asset, ticker in DYDX_TICKERS.items():
        stored = 0
        before_height = None
        pages = 0
        oldest_iso = None
        try:
            while pages < 60:
                pages += 1
                url = f"{DYDX_INDEXER}/{ticker}?limit=1000"
                if before_height is not None:
                    url += f"&effectiveBeforeOrAtHeight={before_height}"
                for attempt in range(4):
                    try:
                        resp = _get_json(url)
                        break
                    except urllib.error.HTTPError as e:
                        if e.code == 429:
                            time.sleep(1.5 * (attempt + 1)); continue
                        if e.code == 404:
                            resp = {"historicalFunding": []}; break
                        raise
                else:
                    print(f"  dYdX {asset}: rate-limited out, partial"); break
                recs = resp.get("historicalFunding", resp) if isinstance(resp, dict) else resp
                if not recs:
                    break
                rows = []
                min_height = None
                for d in recs:
                    eff = d["effectiveAt"]
                    t = int(datetime.fromisoformat(eff.replace("Z", "+00:00"))
                            .timestamp() * 1000)
                    th = (t // HOUR_MS) * HOUR_MS
                    oldest_iso = eff
                    h = int(d["effectiveAtHeight"])
                    min_height = h if min_height is None else min(min_height, h)
                    rows.append((asset, "dydx", th, _iso(th),
                                 float(d["rate"]), "hourly"))
                _store(conn, rows)
                if min_height is None or min_height <= 1:
                    break
                # stop once we've paged past the window start
                if t < START_MS:
                    break
                before_height = min_height - 1
                time.sleep(0.2)
            n = conn.execute("SELECT COUNT(*) FROM funding_rates WHERE "
                             "venue='dydx' AND asset=?", (asset,)).fetchone()[0]
            print(f"  dYdX {asset:<6} pages={pages:<3} db_rows={n} oldest={oldest_iso}")
        except Exception as e:
            print(f"  dYdX {asset}: ERROR {e!r}")


def main():
    print(f"[cycle56] research DB: {RESEARCH_DB}")
    print(f"[cycle56] window: {_iso(START_MS)} .. {_iso(NOW_MS)}")
    conn = init_db()
    try:
        print("=== Hyperliquid (hourly funding) ===")
        collect_hyperliquid(conn)
        print("=== dYdX v4 (hourly funding) ===")
        collect_dydx(conn)
        print("=== DONE ===")
        for row in conn.execute(
                "SELECT venue, COUNT(DISTINCT asset), COUNT(*), "
                "MIN(datetime), MAX(datetime) FROM funding_rates GROUP BY venue"):
            print(" ", row)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
