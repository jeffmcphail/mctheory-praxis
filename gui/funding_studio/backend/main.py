"""
gui/funding_studio/backend/main.py
==================================
Cycle 54: Funding Studio backend (FastAPI + WebSocket) -- Engine 7 control.

A thin control surface over the VERIFIED executor + the shared funding_replay
engine. It implements ZERO trade logic: every booked decision comes from
engines.funding_executor / engines.funding_replay. The backend only
starts/stops/streams sessions and reads back what the executor booked.

No execution-venue interaction; the only side effects are local SQLite writes
performed by the driven executor. (A safety-belt grep over this backend
expects no exchange-client imports or outbound order calls.)

Run from the praxis project root:
    python start_funding_studio.py
(or, from this backend dir:  uvicorn main:app --port 8002 --reload)
"""
import sys
from pathlib import Path

# Project root on sys.path so engines/ + servers/ + src/ import (gui/
# funding_studio/backend/main.py -> parents[3] == repo root).
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import sqlite3

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from engines.trading_session import (
    DB_PATH as MAIN_DB, get_session, list_sessions,
    mark_interrupted_running_sessions,
)
from servers.praxis_mcp.tools.meta import collector_health_snapshot

from controller import SessionController, compute_rollup

app = FastAPI(title="Funding Studio (Engine 7 control)", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

controller = SessionController()


@app.on_event("startup")
def _startup() -> None:
    # Cross-path point 1: sweep ONLY GUI-owned 'running' sessions orphaned by a
    # prior backend process. NEVER touch scheduled sessions (different owner; a
    # mid-run scheduled fire is briefly 'running').
    n = mark_interrupted_running_sessions(MAIN_DB, trigger_source="gui")
    if n:
        print(f"[startup] marked {n} orphaned GUI session(s) interrupted")


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    mode: str                          # "paper_live" | "paper_replay"
    config_overrides: dict = {}
    replay_start: str | None = None    # paper_replay (default = full OOS window)
    replay_end: str | None = None
    gate: float = 0.70                 # paper_replay alert gate
    assets: list[str] | None = None    # paper_replay asset subset
    notes: str | None = None


# ---------------------------------------------------------------------------
# Mode-aware read helpers (guardrail D)
# ---------------------------------------------------------------------------

def _require(sid: str) -> dict:
    row = get_session(MAIN_DB, session_id=sid)
    if not row:
        raise HTTPException(404, "no such session")
    return row


def _session_db(row: dict) -> str:
    """The DB holding a session's booked rows: replay -> its isolated harness
    DB; live -> the MAIN db (R3 structural isolation)."""
    if row["mode"] == "paper_replay":
        return row["harness_db_path"]
    return str(MAIN_DB)


def _read_rows(row: dict, sql: str, params: tuple) -> list[dict]:
    # Replay harness DBs are always opened read-only; the live MAIN db is read
    # with a normal connection (the executor owns its writes).
    if row["mode"] == "paper_replay":
        conn = sqlite3.connect(f"file:{_session_db(row)}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(_session_db(row))
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/sessions")
async def create_session_ep(req: CreateSessionRequest):
    """Create + start a session. Replay drives the shared funding_replay engine;
    live loops the production executor on the MAIN db."""
    if req.mode == "paper_live":
        sid = await controller.start_live(req.config_overrides, req.notes)
    elif req.mode == "paper_replay":
        sid = await controller.start_replay(
            req.replay_start, req.replay_end, req.gate, req.assets,
            req.config_overrides, req.notes)
    else:
        raise HTTPException(400, f"invalid mode {req.mode!r} "
                                 f"(expected paper_live | paper_replay)")
    return {"session_id": sid, "mode": req.mode, "status": "running"}


@app.post("/api/sessions/{sid}/stop")
async def stop_session_ep(sid: str):
    sess = controller.get(sid)
    if not sess:
        row = _require(sid)
        return {"session_id": sid, "status": row["status"],
                "note": "not running (already ended)"}
    sess.stop_event.set()
    return {"session_id": sid, "status": "stopping"}


@app.get("/api/sessions")
def list_sessions_ep(limit: int = 200):
    return list_sessions(MAIN_DB, limit=limit)


@app.get("/api/sessions/{sid}")
def get_session_ep(sid: str):
    row = _require(sid)
    rollup = None
    if row.get("pnl_rollup_json"):
        try:
            rollup = json.loads(row["pnl_rollup_json"])
        except Exception:
            rollup = None
    if rollup is None:
        # Live session without a persisted rollup yet: compute it live.
        try:
            rollup = compute_rollup(_session_db(row), sid)
        except Exception:
            rollup = None
    return {"session": row, "rollup": rollup}


@app.get("/api/sessions/{sid}/trades")
def trades_ep(sid: str):
    row = _require(sid)
    return _read_rows(
        row,
        "SELECT * FROM paper_trades WHERE session_id=? ORDER BY signal_timestamp",
        (sid,))


@app.get("/api/sessions/{sid}/exits")
def exits_ep(sid: str):
    row = _require(sid)
    return _read_rows(
        row,
        "SELECT * FROM paper_position_exits WHERE session_id=? "
        "ORDER BY signal_timestamp",
        (sid,))


@app.get("/api/sessions/{sid}/positions")
def positions_ep(sid: str):
    row = _require(sid)
    return _read_rows(
        row,
        "SELECT p.* FROM paper_trades p WHERE p.session_id=? "
        "AND p.decision='enter' AND NOT EXISTS ("
        "  SELECT 1 FROM paper_position_exits x "
        "  WHERE x.asset=p.asset AND x.signal_timestamp=p.signal_timestamp) "
        "ORDER BY p.signal_timestamp",
        (sid,))


@app.get("/api/health")
def health_ep():
    """Proxy the shared collector-health snapshot (primary DB) so the dashboard
    can show collector freshness."""
    return collector_health_snapshot(MAIN_DB)


def _read_global(sql: str, limit: int) -> list[dict]:
    """Read-only query against MAIN for GLOBAL (non-session-scoped) monitor
    tables. Tolerates a missing table -> []. This is reporting over rows the
    monitor wrote; the backend computes no signals here."""
    conn = sqlite3.connect(f"file:{MAIN_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, (limit,)).fetchall()]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


@app.get("/api/signals")
def signals_ep(limit: int = 60):
    """Recent `funding_signals` — the monitor's per-window, per-asset output.
    GLOBAL (funding_signals has no session_id): this powers the live-monitor
    regime view (how close each asset is to the gate + whether its funding is
    favorable), independent of any session. Ordered newest-first so the client
    can take the latest row per asset plus a little history. Read-only;
    pre-Cycle-53 rows may carry NULL `min_pct_positive` (render as em-dash)."""
    return _read_global(
        "SELECT asset, timestamp, datetime, p_profitable, above_gate, "
        "gate_threshold, ann_rate, pct_positive, hold_days, min_funding_ann, "
        "min_pct_positive FROM funding_signals "
        "ORDER BY timestamp DESC, asset LIMIT ?", limit)


@app.get("/api/alerts")
def alerts_ep(limit: int = 60):
    """The `funding_alerts` ledger (one row per fired alert) — GLOBAL,
    read-only. Empty in a sit-out regime; an empty list is itself the
    sit-out signal."""
    return _read_global(
        "SELECT asset, timestamp, datetime, alerted_at, p_profitable, "
        "gate_threshold, monitor_version FROM funding_alerts "
        "ORDER BY timestamp DESC, asset LIMIT ?", limit)


@app.websocket("/api/ws/sessions/{sid}")
async def ws_session(ws: WebSocket, sid: str):
    await ws.accept()
    sess = controller.get(sid)
    if sess is None:
        # Not running: surface the terminal state from the DB, then close.
        row = get_session(MAIN_DB, session_id=sid)
        if row:
            rollup = (json.loads(row["pnl_rollup_json"])
                      if row.get("pnl_rollup_json") else None)
            await _safe_send(ws, {"type": "state", "phase": row["status"],
                                  "from_db": True, "rollup": rollup})
        await _safe_send(ws, {"type": "closed"})
        await ws.close()
        return
    q = controller.subscribe(sid)
    try:
        if sess.last_state:
            await _safe_send(ws, sess.last_state)
        while True:
            frame = await q.get()
            await _safe_send(ws, frame)
            if frame.get("type") in ("done", "stopped", "error", "closed"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        controller.unsubscribe(sid, q)


async def _safe_send(ws: WebSocket, data: dict) -> None:
    try:
        await ws.send_json(data)
    except Exception:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True,
                reload_dirs=[str(_ROOT / "engines"), str(_ROOT / "gui")])
