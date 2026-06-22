#!/usr/bin/env python3
"""
scripts/funding_regime_alert.py
===============================
Track A (Cycle 58): funding-regime-turn alert.

Carry was PARKED in Cycle 57 (the funding regime went dead: BTC carry
Sharpe 8.60 in 2024 -> 5.26 in 2025 -> 0.58 last-90d, as gross funding
fell 12% -> 5% -> 0.6% annualized). This script is the read-only
"wake me when it's worth re-measuring" tripwire: it watches production
CEX-majors funding on the executor venue and fires a single push alert
when the cross-asset funding level climbs back above a threshold AND
stays there (persistence, not a one-bar spike).

It is deliberately a PROXY. CEX and DeFi funding are correlated; when
CEX majors turn richly positive and sustain, DeFi venues (Hyperliquid
etc., per Cycle 56) follow and tend to run richer. The v1 proxy is
sufficient and cheap; a DeFi-funding-aware version is a later cycle
(needs scheduled DeFi collection + venue-scoping work).

Design (all knobs are params; state defaults shown):
  - READ-ONLY on market data. Reads funding_rates scoped to the SAME
    venue the executor trades (EXECUTION_VENUE), i.e. the Cycle-57-P4
    venue-scoped read. No collectors, no live API calls.
  - SIGNAL: rolling N-day mean of the cross-asset annualized funding
    (averaged over the candidate assets) must exceed THRESHOLD, and
    that condition must hold for M consecutive complete UTC days.
  - EDGE-TRIGGERED + idempotent: fires once per "turn episode" (the
    rising edge), deduped via a small funding_regime_alerts ledger
    (same idea as the funding_alerts ledger). Re-running in the same
    episode is a no-op.
  - Fires via the existing alert channel (PRAXIS_ALERT_URL, ntfy.sh
    backend; TEAMS_WEBHOOK_URL still accepted as a legacy fallback).
  - ASCII-only stdout (Windows cp1252-safe per the collector
    convention). Honest exit code: non-zero only on a real failure
    (DB read error, or an alert POST that failed after a fire).

Usage:
    # Dry run: compute + print what WOULD fire, never POST, never write
    python scripts/funding_regime_alert.py

    # Live: POST the alert if a fresh turn is detected, record in ledger
    python scripts/funding_regime_alert.py --alert

    # Tune the regime definition
    python scripts/funding_regime_alert.py --n-days 7 --m-days 5 \
        --threshold-ann-pct 5.0 --assets BTC,ETH,SOL,XRP,ADA,AVAX
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

# Single source of truth for the venue the executor settles on, so the
# regime proxy reads the SAME funding tape the strategy would trade
# (Cycle 57 P4 venue-scoping discipline).
from engines.funding_executor import EXECUTION_VENUE

# ── Constants / defaults ────────────────────────────────────────────────────

# Candidate carry majors that exist in production funding_rates on the
# executor venue. (Cycle 57 flagged LINK/LTC/DOGE/UNI as additionally
# carry-viable, but those are not collected in production funding_rates
# yet; v1 is the CEX-majors proxy only.)
DEFAULT_ASSETS = ["BTC", "ETH", "SOL", "XRP", "ADA", "AVAX"]

# Rolling-mean window (days) for the regime level.
DEFAULT_N_DAYS = 7
# Persistence: the regime level must clear threshold for this many
# consecutive complete UTC days before we call it a sustained turn.
DEFAULT_M_DAYS = 5
# Threshold on the cross-asset N-day mean annualized funding (%).
# Anchor (Cycle 57): 2024 ~= 12% ann (Sharpe 8.6); 2025 ~= 5% (Sharpe
# 5.3); last-90d ~= 0.6% (Sharpe 0.6). 5.0% ~ the "2025, clearly worth
# re-measuring" regime. A re-measure trigger, NOT a deploy threshold.
DEFAULT_THRESHOLD_ANN_PCT = 5.0
# How much history to read (must comfortably exceed N + M).
DEFAULT_LOOKBACK_DAYS = 45
# Warn if the freshest funding day is older than this (collector lag).
STALE_WARN_DAYS = 3

DEFAULT_DB = str(Path(__file__).resolve().parent.parent / "data" / "crypto_data.db")

# rate-per-8h -> annualized percent: 3 payments/day * 365 days * 100.
ANNUALIZE = 3 * 365 * 100

ALERT_LEDGER_DDL = """
CREATE TABLE IF NOT EXISTS funding_regime_alerts (
    detected_date     TEXT    NOT NULL,   -- UTC date the turn episode began (PK)
    alerted_at        TEXT    NOT NULL,   -- ISO 8601 +00:00 when this row was written
    regime_level_ann  REAL    NOT NULL,   -- cross-asset N-day mean annualized %
    threshold_ann     REAL    NOT NULL,
    n_days            INTEGER NOT NULL,
    m_days            INTEGER NOT NULL,
    venue             TEXT    NOT NULL,
    assets            TEXT    NOT NULL,    -- comma-joined asset set
    PRIMARY KEY (detected_date)
)
"""


# ── Read (venue-scoped, read-only) ──────────────────────────────────────────

def read_funding(db_path: str, assets: list[str], venue: str,
                 lookback_days: int) -> pd.DataFrame:
    """Read funding_rates for `assets` on `venue` over the lookback window.

    Returns a wide DataFrame: index = UTC calendar date, one column per
    asset holding that day's MEAN funding rate (per-8h, not annualized).
    Read-only connection (mode=ro). Missing assets simply yield no column.
    """
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    since_ms = now_ms - (lookback_days + 2) * 86_400_000

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        frames = {}
        for asset in assets:
            rows = conn.execute(
                "SELECT timestamp, funding_rate FROM funding_rates "
                "WHERE asset = ? AND venue = ? "
                "AND timestamp >= ? AND timestamp <= ? "
                "ORDER BY timestamp",
                (asset, venue, since_ms, now_ms),
            ).fetchall()
            if not rows:
                continue
            s = pd.DataFrame(rows, columns=["timestamp", "rate"])
            s["timestamp"] = pd.to_datetime(s["timestamp"], unit="ms", utc=True)
            s = s.dropna(subset=["rate"]).set_index("timestamp").sort_index()
            # Mean of the (up to 3) funding payments on each UTC calendar day.
            daily = s["rate"].resample("1D").mean()
            frames[asset] = daily
    finally:
        conn.close()

    if not frames:
        return pd.DataFrame()
    wide = pd.DataFrame(frames).sort_index()
    return wide


# ── Signal ──────────────────────────────────────────────────────────────────

def compute_regime(wide: pd.DataFrame, n_days: int, m_days: int,
                   threshold_ann_pct: float) -> dict:
    """Turn the wide daily-rate frame into the regime decision.

    - Drops the current (possibly partial) UTC day so the read is
      deterministic and evaluated only on complete days.
    - regime_level = rolling N-day mean of the cross-asset daily mean,
      annualized to percent.
    - sustained = regime_level > threshold for M consecutive days.
    - Fires on the first day of the current sustained run (episode start).

    Returns a dict describing the as-of evaluation (does not POST).
    """
    out = {
        "ok": False, "reason": "", "as_of": None, "fire": False,
        "episode_start": None, "regime_level_ann": None,
        "per_asset_ann": {}, "n_days": n_days, "m_days": m_days,
        "threshold_ann_pct": threshold_ann_pct, "stale": False,
        "latest_data_date": None,
    }
    if wide.empty:
        out["reason"] = "no funding rows for any candidate asset on venue"
        return out

    today = pd.Timestamp.now(tz="UTC").normalize()
    # Evaluate on complete days only: drop today's partial bucket.
    complete = wide[wide.index < today]
    if complete.empty:
        out["reason"] = "no complete UTC days in window"
        return out

    out["latest_data_date"] = complete.index[-1].strftime("%Y-%m-%d")
    days_stale = (today - complete.index[-1]).days
    out["stale"] = days_stale > STALE_WARN_DAYS

    # Cross-asset daily mean rate (skip assets missing that day), annualized.
    cross = complete.mean(axis=1, skipna=True)
    cross_ann = cross * ANNUALIZE

    regime_level = cross_ann.rolling(n_days, min_periods=n_days).mean()
    regime_on = regime_level > threshold_ann_pct  # NaN -> False
    sustained = (regime_on.astype(int)
                 .rolling(m_days, min_periods=m_days).sum() == m_days)

    if len(sustained) == 0 or pd.isna(regime_level.iloc[-1]):
        out["ok"] = True
        out["reason"] = "insufficient complete history for N+M window"
        out["as_of"] = complete.index[-1].strftime("%Y-%m-%d")
        return out

    as_of = sustained.index[-1]
    out["ok"] = True
    out["as_of"] = as_of.strftime("%Y-%m-%d")
    out["regime_level_ann"] = float(regime_level.iloc[-1])
    # Per-asset N-day mean annualized (for the alert body).
    per_asset = (complete.rolling(n_days, min_periods=n_days).mean().iloc[-1]
                 * ANNUALIZE)
    out["per_asset_ann"] = {a: float(v) for a, v in per_asset.items()
                            if pd.notna(v)}

    if bool(sustained.iloc[-1]):
        # Walk back over the contiguous sustained run to find its start.
        i = len(sustained) - 1
        while i - 1 >= 0 and bool(sustained.iloc[i - 1]):
            i -= 1
        out["fire"] = True
        out["episode_start"] = sustained.index[i].strftime("%Y-%m-%d")
    return out


# ── Alert (existing channel) ────────────────────────────────────────────────

def resolve_alert_url() -> tuple[str, str]:
    """PRAXIS_ALERT_URL, falling back to legacy TEAMS_WEBHOOK_URL."""
    url = os.getenv("PRAXIS_ALERT_URL", "").strip()
    if url:
        return url, "PRAXIS_ALERT_URL"
    url = os.getenv("TEAMS_WEBHOOK_URL", "").strip()
    if url:
        return url, "TEAMS_WEBHOOK_URL (legacy fallback)"
    return "", ""


def build_alert_body(res: dict, venue: str, assets: list[str]) -> str:
    """ASCII markdown body for the ntfy push."""
    per = res.get("per_asset_ann", {})
    per_str = ", ".join(f"{a}={per[a]:+.1f}%" for a in assets if a in per)
    return (
        f"**Regime level (N={res['n_days']}d cross-asset mean):** "
        f"{res['regime_level_ann']:+.2f}% ann\n"
        f"**Threshold:** {res['threshold_ann_pct']:.2f}% ann, "
        f"sustained {res['m_days']}d (since {res['episode_start']})\n"
        f"**Per-asset N-day mean:** {per_str}\n"
        f"**Venue:** {venue}  **Assets:** {','.join(assets)}\n"
        f"**As of:** {res['as_of']}\n"
        f"NOTE: carry parked since Cycle 57 (regime had gone dead). This "
        f"is a RE-MEASURE trigger, not a deploy signal. Re-run the "
        f"Cycle 57 basis-inclusive Sharpe before risking capital."
    )


def post_alert(url: str, body: str) -> tuple[bool, str]:
    """POST an ASCII push to the ntfy.sh-style backend. Returns (ok, msg)."""
    req = urllib.request.Request(
        url, data=body.encode("utf-8"),
        headers={
            "Content-Type": "text/plain; charset=utf-8",
            "Title": "FUNDING REGIME TURN (carry re-measure trigger)",
            "Tags": "rotating_light",
            "Priority": "4",
            "Markdown": "yes",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.status
            txt = resp.read().decode("utf-8", errors="replace")
            return status < 300, f"HTTP {status}: {txt[:160]}"
    except Exception as e:
        return False, f"ERROR: {type(e).__name__}: {e}"[:200]


# ── Ledger (dedup) ──────────────────────────────────────────────────────────

def already_alerted(db_path: str, episode_start: str) -> bool:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(ALERT_LEDGER_DDL)
        row = conn.execute(
            "SELECT 1 FROM funding_regime_alerts WHERE detected_date = ?",
            (episode_start,),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def record_alert(db_path: str, res: dict, venue: str, assets: list[str]) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(ALERT_LEDGER_DDL)
        conn.execute(
            "INSERT OR IGNORE INTO funding_regime_alerts "
            "(detected_date, alerted_at, regime_level_ann, threshold_ann, "
            " n_days, m_days, venue, assets) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (res["episode_start"],
             datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
             float(res["regime_level_ann"]), float(res["threshold_ann_pct"]),
             int(res["n_days"]), int(res["m_days"]), venue, ",".join(assets)),
        )
        conn.commit()
    finally:
        conn.close()


# ── Orchestration ───────────────────────────────────────────────────────────

def run_once(args) -> int:
    assets = [a.strip().upper() for a in args.assets.split(",") if a.strip()] \
        if args.assets else list(DEFAULT_ASSETS)

    print("=" * 64)
    print("FUNDING REGIME TURN ALERT (Track A)")
    print(f"  venue={EXECUTION_VENUE}  assets={','.join(assets)}")
    print(f"  N={args.n_days}d mean  M={args.m_days}d sustained  "
          f"threshold={args.threshold_ann_pct:.2f}% ann")
    print("=" * 64)

    try:
        wide = read_funding(args.db, assets, EXECUTION_VENUE, args.lookback_days)
    except Exception as e:
        print(f"  ERROR reading funding_rates: {type(e).__name__}: {e}")
        return 1

    res = compute_regime(wide, args.n_days, args.m_days, args.threshold_ann_pct)
    if not res["ok"]:
        print(f"  No signal: {res['reason']}")
        return 0

    missing = [a for a in assets if a not in wide.columns]
    if missing:
        print(f"  NOTE: no {EXECUTION_VENUE} funding rows for: "
              f"{','.join(missing)} (excluded)")
    print(f"  As of (last complete day): {res['as_of']}  "
          f"(latest data: {res['latest_data_date']})")
    if res["stale"]:
        print(f"  WARN: freshest funding day is stale (>{STALE_WARN_DAYS}d "
              f"behind today); regime read is lagged.")
    if res["regime_level_ann"] is not None:
        print(f"  Regime level (N={args.n_days}d cross-asset mean): "
              f"{res['regime_level_ann']:+.2f}% ann "
              f"(threshold {args.threshold_ann_pct:+.2f}%)")
        per = res["per_asset_ann"]
        print("  Per-asset N-day mean ann: " +
              ", ".join(f"{a}={per[a]:+.1f}%" for a in assets if a in per))

    if not res["fire"]:
        print("  REGIME: below sustained threshold -> no turn. (carry stays "
              "parked)")
        return 0

    print(f"  REGIME TURN DETECTED: sustained >= {args.m_days}d since "
          f"{res['episode_start']}")

    if not args.alert:
        print("  [dry-run] would fire alert (pass --alert to POST + record).")
        return 0

    if already_alerted(args.db, res["episode_start"]):
        print(f"  Already alerted for this episode "
              f"(detected_date={res['episode_start']}); no-op.")
        return 0

    url, src = resolve_alert_url()
    if not url:
        print("  WARN: neither PRAXIS_ALERT_URL nor TEAMS_WEBHOOK_URL set; "
              "--alert is a no-op (no ledger row written).")
        return 0

    print(f"  Webhook URL source: {src}")
    body = build_alert_body(res, EXECUTION_VENUE, assets)
    ok, response = post_alert(url, body)
    if not ok:
        print(f"  ALERT FAILED: {response}")
        return 1
    record_alert(args.db, res, EXECUTION_VENUE, assets)
    print(f"  ALERT delivered + recorded ({response})")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Funding-regime-turn alert (Track A, Cycle 58)")
    p.add_argument("--assets", default=None,
                   help="Comma-separated assets (default: "
                        "BTC,ETH,SOL,XRP,ADA,AVAX)")
    p.add_argument("--n-days", type=int, default=DEFAULT_N_DAYS,
                   help=f"Rolling-mean window days (default {DEFAULT_N_DAYS})")
    p.add_argument("--m-days", type=int, default=DEFAULT_M_DAYS,
                   help=f"Persistence days (default {DEFAULT_M_DAYS})")
    p.add_argument("--threshold-ann-pct", type=float,
                   default=DEFAULT_THRESHOLD_ANN_PCT,
                   help="Annualized %% threshold on the cross-asset N-day "
                        f"mean (default {DEFAULT_THRESHOLD_ANN_PCT})")
    p.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS,
                   help=f"History to read (default {DEFAULT_LOOKBACK_DAYS})")
    p.add_argument("--db", default=DEFAULT_DB, help="crypto_data.db path")
    p.add_argument("--alert", action="store_true",
                   help="Actually POST the alert + write the ledger row on a "
                        "fresh turn (default: dry-run, no POST, no write).")
    args = p.parse_args()
    return run_once(args)


if __name__ == "__main__":
    sys.exit(main())
