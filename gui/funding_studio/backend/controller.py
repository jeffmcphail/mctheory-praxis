"""
gui/funding_studio/backend/controller.py
========================================
Cycle 54: the Funding Studio session controller.

DRIVES the verified executor + the shared funding_replay engine; implements
ZERO trade logic (no signals, sizing, or gating live here):
  - a LIVE session loops the production FundingExecutor.run_once() on the MAIN
    db with the real clock (default now_func), polling every LIVE_POLL_SECONDS;
  - a REPLAY session runs the shared funding_replay engine (regenerate_predictions
    -> build_harness -> run_executor) against an ISOLATED per-session harness DB
    with an injected clock settled past all in-window holds.

Session lifecycle (create/close) is delegated to engines.trading_session -- the
controller does not grow its own. No execution-venue interaction anywhere; the
only side effects are local SQLite writes performed by the executor it drives.

Known cross-path interaction (documented, not engineered around this cycle):
a GUI live session and the PraxisFundingExecutor scheduled task both run_once()
on the MAIN db. For paper this is benign -- the paper_trades PK + INSERT OR
IGNORE dedups each alert to exactly one row -- though session attribution is
racy (whichever executor wins). This becomes a HARD prerequisite to resolve
(retire/coexist the scheduled task) before the real-money cycle: two live
executors must not drive one real account.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

from engines.funding_executor import FundingExecutor, EXECUTOR_VERSION
from engines.funding_replay import (
    regenerate_predictions, build_harness, run_executor, OOS_START, OOS_END,
)
from engines.trading_session import (
    DB_PATH as MAIN_DB, create_session, close_session, new_session_id,
)

REPO = Path(__file__).resolve().parents[3]
SESSIONS_DIR = REPO / "outputs" / "funding_studio_sessions"

LIVE_POLL_SECONDS = 60

# Replay settles the injected clock PAST every in-window hold so each position
# opened in-window reaches its exit and the rollup is COMPLETE P&L (not "some
# still open"). 30d comfortably exceeds the 14d max hold in the config grid.
# (The now=window_end "watch it stream with positions still open" semantics are
# the stepped-clock view, deferred to 54b/c.)
REPLAY_SETTLE_DAYS = 30


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def compute_rollup(db_path: Path | str, session_id: str) -> dict:
    """Aggregate a session's booked rows into a P&L headline. Read-only; this
    is reporting over what the executor booked, not trade logic."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        entries = conn.execute(
            "SELECT COUNT(*) FROM paper_trades "
            "WHERE session_id=? AND decision='enter'", (session_id,)).fetchone()[0]
        skips = conn.execute(
            "SELECT COUNT(*) FROM paper_trades "
            "WHERE session_id=? AND decision='skip'", (session_id,)).fetchone()[0]
        n_exits, net, funding, tc = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(net_pnl_usd), 0.0), "
            "       COALESCE(SUM(funding_payments_usd), 0.0), "
            "       COALESCE(SUM(tc_entry_usd + tc_exit_usd), 0.0) "
            "FROM paper_position_exits WHERE session_id=?", (session_id,)).fetchone()
        by_asset = {
            a: {"exits": n, "net_pnl_usd": round(p, 6)}
            for a, n, p in conn.execute(
                "SELECT asset, COUNT(*), COALESCE(SUM(net_pnl_usd), 0.0) "
                "FROM paper_position_exits WHERE session_id=? GROUP BY asset "
                "ORDER BY asset", (session_id,)).fetchall()
        }
        return {
            "entries": entries,
            "skips": skips,
            "exits": n_exits,
            "net_pnl_usd": round(net, 6),
            "funding_payments_usd": round(funding, 6),
            "transaction_costs_usd": round(tc, 6),
            "by_asset": by_asset,
        }
    finally:
        conn.close()


def compute_equity_series(db_path: Path | str, session_id: str) -> list[dict]:
    """Realized-P&L equity curve: cumulative net_pnl_usd stepped at each exit
    timestamp. The executor books P&L only at exit (no intra-hold MTM), so the
    curve is a step function. Grouped by exit_timestamp so points are unique on
    the time axis (multiple assets can exit the same instant); the terminal
    cum_pnl_usd ties out to compute_rollup's net_pnl_usd. Read-only -- reporting
    over booked exits, not trade logic."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT exit_timestamp, MIN(exit_datetime) AS dt, "
            "       SUM(net_pnl_usd) AS step, COUNT(*) AS n "
            "FROM paper_position_exits WHERE session_id=? "
            "GROUP BY exit_timestamp ORDER BY exit_timestamp", (session_id,)).fetchall()
    finally:
        conn.close()
    series, cum = [], 0.0
    for ts, dt, step, n in rows:
        cum += step
        series.append({
            "exit_timestamp": int(ts),
            "exit_datetime": dt,
            "exits_at_step": int(n),
            "step_pnl_usd": round(step, 6),
            "cum_pnl_usd": round(cum, 6),
        })
    return series


class RunningSession:
    def __init__(self, session_id: str, mode: str):
        self.id = session_id
        self.mode = mode
        self.task: asyncio.Task | None = None
        self.stop_event = asyncio.Event()
        self.subscribers: set[asyncio.Queue] = set()
        self.last_state: dict | None = None


class SessionController:
    """Owns running GUI sessions (in-memory registry keyed by session_id)."""

    def __init__(self) -> None:
        self._sessions: dict[str, RunningSession] = {}
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- streaming pub/sub ----
    def subscribe(self, session_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        sess = self._sessions.get(session_id)
        if sess:
            sess.subscribers.add(q)
        return q

    def unsubscribe(self, session_id: str, q: asyncio.Queue) -> None:
        sess = self._sessions.get(session_id)
        if sess:
            sess.subscribers.discard(q)

    def get(self, session_id: str) -> RunningSession | None:
        return self._sessions.get(session_id)

    async def _emit(self, sess: RunningSession, frame: dict) -> None:
        frame = {"session_id": sess.id, "ts": _utcnow_iso(), **frame}
        sess.last_state = frame
        for q in list(sess.subscribers):
            q.put_nowait(frame)

    # ---- live ----
    async def start_live(self, config_overrides: dict | None,
                         notes: str | None) -> str:
        sid = new_session_id()
        # One executor just to snapshot the config in effect for the row.
        snap = FundingExecutor(db_path=MAIN_DB, session_id=sid,
                               defaults_override=config_overrides or None)
        config_snapshot = {
            **snap.config,
            "enforce_config_gate": snap.enforce_config_gate,
            "kill_switch_on": snap.kill_switch_on,
        }
        create_session(MAIN_DB, mode="paper_live", trigger_source="gui",
                       session_id=sid,
                       config_json=json.dumps(config_snapshot, sort_keys=True),
                       executor_version=EXECUTOR_VERSION, status="running",
                       notes=notes)
        sess = RunningSession(sid, "paper_live")
        self._sessions[sid] = sess
        sess.task = asyncio.create_task(self._live_loop(sess, config_overrides))
        return sid

    async def _live_loop(self, sess: RunningSession,
                         config_overrides: dict | None) -> None:
        loop = asyncio.get_running_loop()
        await self._emit(sess, {"type": "state", "phase": "started",
                                "mode": "paper_live"})
        try:
            while not sess.stop_event.is_set():
                try:
                    # Fresh executor each poll: re-reads the kill-switch env +
                    # config so a long-lived live session never runs on a stale
                    # constant (long-lived-collector lesson). All booked state
                    # lives in the DB, so reconstruction is free of side effects.
                    executor = FundingExecutor(
                        db_path=MAIN_DB, session_id=sess.id,
                        defaults_override=config_overrides or None)
                    summary = await loop.run_in_executor(None, executor.run_once)
                    phase = ("idle/holding"
                             if summary["entered"] == 0 and summary["exited"] == 0
                             else "active")
                    await self._emit(sess, {
                        "type": "state", "phase": phase,
                        "last_run_at": _utcnow_iso(), "summary": summary,
                    })
                except Exception as exc:
                    await self._emit(sess, {
                        "type": "error", "message": str(exc),
                        "trace": traceback.format_exc()[-800:]})
                    close_session(MAIN_DB, sess.id, status="error")
                    return
                # Wait for the poll interval, but wake immediately on stop.
                try:
                    await asyncio.wait_for(sess.stop_event.wait(),
                                           timeout=LIVE_POLL_SECONDS)
                except asyncio.TimeoutError:
                    pass
            close_session(MAIN_DB, sess.id, status="stopped")
            await self._emit(sess, {"type": "stopped"})
        finally:
            self._sessions.pop(sess.id, None)

    # ---- replay ----
    async def start_replay(self, oos_start: str | None, oos_end: str | None,
                           gate: float, assets: list | None,
                           config_overrides: dict | None,
                           notes: str | None) -> str:
        sid = new_session_id()
        oos_start = oos_start or OOS_START
        oos_end = oos_end or OOS_END
        harness_db = SESSIONS_DIR / f"{sid}.db"
        config_snapshot = {
            "gate": gate, "assets": assets,
            "oos_start": oos_start, "oos_end": oos_end,
            **(config_overrides or {}),
        }
        create_session(MAIN_DB, mode="paper_replay", trigger_source="gui",
                       session_id=sid,
                       config_json=json.dumps(config_snapshot, sort_keys=True),
                       executor_version=EXECUTOR_VERSION, status="running",
                       replay_start=oos_start, replay_end=oos_end,
                       harness_db_path=str(harness_db), notes=notes)
        sess = RunningSession(sid, "paper_replay")
        self._sessions[sid] = sess
        sess.task = asyncio.create_task(
            self._replay_task(sess, harness_db, oos_start, oos_end, gate, assets))
        return sid

    def _stopped_midway(self, sess: RunningSession) -> bool:
        if sess.stop_event.is_set():
            close_session(MAIN_DB, sess.id, status="stopped")
            return True
        return False

    async def _replay_task(self, sess: RunningSession, harness_db: Path,
                           oos_start: str, oos_end: str, gate: float,
                           assets: list | None) -> None:
        loop = asyncio.get_running_loop()
        try:
            await self._emit(sess, {"type": "state", "phase": "regenerating",
                                    "window": [oos_start, oos_end], "gate": gate})
            preds, fseries = await loop.run_in_executor(
                None, lambda: regenerate_predictions(
                    assets=assets, oos_start=oos_start, oos_end=oos_end))
            if self._stopped_midway(sess):
                await self._emit(sess, {"type": "stopped"})
                return

            await self._emit(sess, {"type": "state", "phase": "building_harness"})
            n_alerts = await loop.run_in_executor(
                None, lambda: build_harness(harness_db, preds, fseries, gate,
                                            monitor_gate=True))
            if self._stopped_midway(sess):
                await self._emit(sess, {"type": "stopped"})
                return

            await self._emit(sess, {"type": "state", "phase": "running_executor",
                                    "alerts": n_alerts})
            end_dt = datetime.strptime(oos_end, "%Y-%m-%d").replace(
                tzinfo=timezone.utc)
            settle = end_dt + timedelta(days=REPLAY_SETTLE_DAYS)
            summary = await loop.run_in_executor(
                None, lambda: run_executor(harness_db, True, sess.id, settle))

            rollup = await loop.run_in_executor(
                None, lambda: compute_rollup(harness_db, sess.id))
            close_session(MAIN_DB, sess.id, status="completed",
                          pnl_rollup_json=json.dumps(rollup, sort_keys=True))
            await self._emit(sess, {"type": "done", "summary": summary,
                                    "rollup": rollup})
        except Exception as exc:
            close_session(MAIN_DB, sess.id, status="error")
            await self._emit(sess, {"type": "error", "message": str(exc),
                                    "trace": traceback.format_exc()[-1200:]})
        finally:
            self._sessions.pop(sess.id, None)
