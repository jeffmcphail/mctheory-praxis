"""
scripts/cycle54_session_threading_smoke.py
===========================================
Cycle 54 verification #2: prove the executor threads session_id onto BOTH
paper tables, the None-session write is refused (fail-loud, since INSERT OR
IGNORE would otherwise silently drop a NOT NULL violation), and the ambient
session helpers write a well-formed trading_sessions row.

Self-contained + throwaway: builds harness DBs in an OS temp dir (zero repo
footprint), injects one config-gate-passing alert/signal + funding events,
runs the executor with a session_id + injected post-window clock, asserts the
booked enter row AND the exit row carry that session_id. Tears everything
down. No execution-venue interaction; the only side effects are local SQLite
writes to throwaway temp DBs.
"""
from __future__ import annotations

import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from engines.funding_executor import FundingExecutor  # noqa: E402
from engines.trading_session import (  # noqa: E402
    create_session, close_session, get_session)

SESSION_ID = "cycle54-smoke-sess"
REPLAY_NOW = datetime(2025, 7, 1, tzinfo=timezone.utc)   # well after the hold window
MS_PER_DAY = 86_400_000

RELAXED = {
    "max_concurrent_positions_per_asset": 1_000_000,
    "max_total_notional_usd": 1e15,
    "max_notional_per_asset_usd": 500.0,
    "max_signal_age_seconds": 10 ** 15,
    "max_daily_loss_usd": 1e15,
    "max_daily_loss_pct": 1e9,
}

# Harness DDL incl session_id NOT NULL on the paper tables (mirrors the
# Cycle 54 migrations + the funding_replay harness DDL).
DDL = {
    "funding_rates":
        "CREATE TABLE funding_rates (asset TEXT NOT NULL, venue TEXT NOT NULL, "
        "timestamp INTEGER NOT NULL, datetime TEXT NOT NULL, funding_rate REAL, "
        "PRIMARY KEY (asset, venue, timestamp))",
    "funding_signals":
        "CREATE TABLE funding_signals (asset TEXT NOT NULL, timestamp INTEGER NOT NULL, "
        "datetime TEXT NOT NULL, p_profitable REAL NOT NULL, above_gate INTEGER NOT NULL, "
        "above_gate_050 INTEGER NOT NULL, gate_threshold REAL NOT NULL, best_config_id TEXT, "
        "hold_days INTEGER, min_funding_ann REAL, expected_return REAL, ann_rate REAL, "
        "basis_pct REAL, pct_positive REAL, min_pct_positive REAL, base_rate REAL, "
        "features_json TEXT, monitor_version TEXT NOT NULL, PRIMARY KEY (asset, timestamp))",
    "funding_alerts":
        "CREATE TABLE funding_alerts (asset TEXT NOT NULL, timestamp INTEGER NOT NULL, "
        "datetime TEXT NOT NULL, alerted_at TEXT NOT NULL, p_profitable REAL NOT NULL, "
        "gate_threshold REAL NOT NULL, monitor_version TEXT NOT NULL, "
        "PRIMARY KEY (asset, timestamp))",
    "paper_trades":
        "CREATE TABLE paper_trades (asset TEXT NOT NULL, signal_timestamp INTEGER NOT NULL, "
        "signal_datetime TEXT NOT NULL, funding_alert_alerted_at TEXT NOT NULL, "
        "decided_at TEXT NOT NULL, decision TEXT NOT NULL, skip_reason TEXT, "
        "intended_direction TEXT, intended_size_usd REAL, p_profitable REAL NOT NULL, "
        "gate_threshold REAL NOT NULL, risk_checks_json TEXT NOT NULL, "
        "executor_version TEXT NOT NULL, hold_days INTEGER, session_id TEXT NOT NULL, "
        "PRIMARY KEY (asset, signal_timestamp))",
    "paper_position_exits":
        "CREATE TABLE paper_position_exits (asset TEXT NOT NULL, signal_timestamp INTEGER NOT NULL, "
        "entry_decided_at TEXT NOT NULL, exit_decided_at TEXT NOT NULL, exit_timestamp INTEGER NOT NULL, "
        "exit_datetime TEXT NOT NULL, hold_days INTEGER NOT NULL, funding_events_count INTEGER NOT NULL, "
        "funding_payments_usd REAL NOT NULL, tc_entry_usd REAL NOT NULL, tc_exit_usd REAL NOT NULL, "
        "net_pnl_usd REAL NOT NULL, notional_usd REAL NOT NULL, direction TEXT NOT NULL, "
        "executor_version TEXT NOT NULL, session_id TEXT NOT NULL, PRIMARY KEY (asset, signal_timestamp))",
    "trading_sessions":
        "CREATE TABLE trading_sessions (session_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, "
        "started_at TEXT, ended_at TEXT, status TEXT NOT NULL, mode TEXT NOT NULL, "
        "trigger_source TEXT NOT NULL, config_json TEXT NOT NULL, executor_version TEXT NOT NULL, "
        "replay_start TEXT, replay_end TEXT, harness_db_path TEXT, pnl_rollup_json TEXT, notes TEXT)",
}

HOLD_DAYS = 3
BASE_TS = int(datetime(2025, 6, 1, tzinfo=timezone.utc).timestamp() * 1000)
BASE_DT = "2025-06-01T00:00:00+00:00"


def build_harness(db: Path) -> None:
    if db.exists():
        db.unlink()
    conn = sqlite3.connect(db)
    for ddl in DDL.values():
        conn.execute(ddl)
    # funding events across the hold window (8h cadence), all positive.
    for k in range(1, HOLD_DAYS * 3 + 1):
        ev = BASE_TS + k * 8 * 3_600_000
        ev_dt = datetime.fromtimestamp(ev / 1000, timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S+00:00")
        conn.execute("INSERT INTO funding_rates VALUES ('SMOKE','binance',?,?,?)",
                     (ev, ev_dt, 0.0002))
    # config-gate-passing signal: ann_rate 30 >= min 5; pct_positive 0.95 >= 0.5.
    conn.execute(
        "INSERT INTO funding_signals (asset,timestamp,datetime,p_profitable,above_gate,"
        "above_gate_050,gate_threshold,best_config_id,hold_days,min_funding_ann,"
        "expected_return,ann_rate,basis_pct,pct_positive,min_pct_positive,base_rate,"
        "features_json,monitor_version) VALUES "
        "('SMOKE',?,?,0.85,1,1,0.70,'fr_smoke',?,5.0,0.0,30.0,0.0,0.95,0.5,0.0,NULL,"
        "'cycle54-smoke')",
        (BASE_TS, BASE_DT, HOLD_DAYS))
    conn.execute(
        "INSERT INTO funding_alerts (asset,timestamp,datetime,alerted_at,p_profitable,"
        "gate_threshold,monitor_version) VALUES ('SMOKE',?,?,?,0.85,0.70,'cycle54-smoke')",
        (BASE_TS, BASE_DT, BASE_DT))
    conn.commit()
    conn.close()


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="cycle54_smoke_"))
    harness = work / "harness.db"
    main_like = work / "main_like.db"
    try:
        print("=" * 70)
        print(" Cycle 54 session-threading smoke")
        print("=" * 70)

        # 1. executor threads session_id onto paper_trades + exits in one run.
        build_harness(harness)
        ex = FundingExecutor(db_path=harness, defaults_override=RELAXED,
                             now_func=lambda: REPLAY_NOW, session_id=SESSION_ID,
                             enforce_config_gate=True)
        summary = ex.run_once()
        conn = sqlite3.connect(harness)
        tr = conn.execute(
            "SELECT decision, session_id FROM paper_trades WHERE asset='SMOKE'"
        ).fetchone()
        xr = conn.execute(
            "SELECT session_id, net_pnl_usd FROM paper_position_exits WHERE asset='SMOKE'"
        ).fetchone()
        conn.close()
        t_ok = tr is not None and tr[0] == "enter" and tr[1] == SESSION_ID
        x_ok = xr is not None and xr[0] == SESSION_ID
        print(f"  summary: {summary}")
        print(f"  paper_trades:         decision={tr[0] if tr else None!r} "
              f"session_id={tr[1] if tr else None!r} -> {'PASS' if t_ok else 'FAIL'}")
        print(f"  paper_position_exits: session_id={xr[0] if xr else None!r} "
              f"net_pnl=${(xr[1] if xr else float('nan')):+.4f} "
              f"-> {'PASS' if x_ok else 'FAIL'}")

        # 2. None-session run_once() is refused (fail-loud).
        build_harness(harness)
        guard_ok = False
        try:
            FundingExecutor(db_path=harness, defaults_override=RELAXED,
                            now_func=lambda: REPLAY_NOW, session_id=None,
                            enforce_config_gate=True).run_once()
        except ValueError:
            guard_ok = True
        print(f"  None-session guard raises ValueError -> "
              f"{'PASS' if guard_ok else 'FAIL'}")

        # 3. ambient-session helpers write a well-formed trading_sessions row.
        c = sqlite3.connect(main_like)
        c.execute(DDL["trading_sessions"])
        c.commit()
        c.close()
        sid = create_session(main_like, mode="paper_live", trigger_source="scheduled",
                             config_json="{}", executor_version="cycle54-smoke",
                             status="running")
        close_session(main_like, sid, status="completed",
                      pnl_rollup_json='{"entered":1}')
        s = get_session(main_like, session_id=sid)
        sess_ok = (s is not None and s["status"] == "completed"
                   and s["mode"] == "paper_live" and s["trigger_source"] == "scheduled"
                   and s["started_at"] is not None and s["ended_at"] is not None
                   and s["pnl_rollup_json"] == '{"entered":1}')
        print(f"  trading_sessions row: status={s['status'] if s else None!r} "
              f"mode={s['mode'] if s else None!r} "
              f"started/ended set={bool(s and s['started_at'] and s['ended_at'])} "
              f"-> {'PASS' if sess_ok else 'FAIL'}")

        all_ok = t_ok and x_ok and guard_ok and sess_ok
        print(f"\n  RESULT: {'ALL PASS' if all_ok else 'FAILURES PRESENT'}")
        return 0 if all_ok else 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
