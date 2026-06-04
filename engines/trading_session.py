"""
engines/trading_session.py
==========================
Cycle 54: first-class trading sessions for Engine 7.

A session is a real entity with a `session_id`; every booked paper row
(paper_trades + paper_position_exits) is attributable to one (schema
contract: session_id NOT NULL going forward -- R2 decision, Cycle 54 RECON).
Sessions are created by:
  - the GUI live/replay controllers (gui/funding_studio),
  - the scheduled bat + CLI via funding_executor.main()'s ambient session,
all writing to the SAME `trading_sessions` table in the main crypto_data.db.

This module ONLY manages session metadata in local SQLite. It runs no trade
logic and has no execution-venue interaction whatsoever; the only side
effects are writes to the local trading_sessions table. The executor
(engines/funding_executor.py) remains the sole writer of the paper tables.

Modes (R3 isolation decision -- structural separation over a query filter):
  paper_live    -> booked rows live in the main crypto_data.db; reads scope
                   to WHERE session_id = ?.
  paper_replay  -> booked rows live in an ISOLATED per-session harness DB so
                   replay's synthetic inputs can never be picked up by the
                   live scheduled executor; the trading_sessions row records
                   `harness_db_path` so the analyze view can find them.
  (real_* modes arrive in the real-money cycle; the GUI renders them with no
   rework because it just shows whatever the executor booked.)

CWD note: the default DB path is __file__-anchored (Cycle 46/47 convention),
so this module is CWD-independent like the executor + funding chain.
"""
from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# __file__-anchored default; mirrors engines/funding_executor.py:DB_PATH.
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "crypto_data.db"

VALID_MODES = ("paper_live", "paper_replay")
VALID_TRIGGER_SOURCES = ("gui", "scheduled", "cli")
VALID_STATUSES = (
    "created", "running", "stopped", "completed", "error", "interrupted",
)

# Columns update_session() is allowed to set (everything mutable post-create).
_UPDATABLE = {
    "started_at", "ended_at", "status", "config_json", "replay_start",
    "replay_end", "harness_db_path", "pnl_rollup_json", "notes",
}


def _utc_now_iso() -> str:
    """ISO-8601 UTC with explicit +00:00, matching the executor's format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def new_session_id() -> str:
    """uuid4 hex -- collision-free and clock-independent."""
    return uuid.uuid4().hex


def create_session(db_path: Path | str = DB_PATH, *,
                   mode: str,
                   trigger_source: str,
                   config_json: str,
                   executor_version: str,
                   status: str = "created",
                   session_id: str | None = None,
                   created_at: str | None = None,
                   started_at: str | None = None,
                   replay_start: str | None = None,
                   replay_end: str | None = None,
                   harness_db_path: str | None = None,
                   notes: str | None = None,
                   now_iso: str | None = None) -> str:
    """Insert a trading_sessions row; return its session_id.

    status defaults to 'created'. Pass status='running' for the ambient
    per-run session (created + started in one shot) -- started_at then
    defaults to the creation instant.

    Raises on a bad enum value, or if trading_sessions is missing (the
    cycle54c migration must have run). Failing loud here is deliberate: a
    run that cannot register its session must NOT proceed to book orphan,
    unattributable rows (guardrail B).
    """
    if mode not in VALID_MODES:
        raise ValueError(f"invalid mode {mode!r}; expected one of {VALID_MODES}")
    if trigger_source not in VALID_TRIGGER_SOURCES:
        raise ValueError(f"invalid trigger_source {trigger_source!r}; "
                         f"expected one of {VALID_TRIGGER_SOURCES}")
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid status {status!r}; "
                         f"expected one of {VALID_STATUSES}")

    sid = session_id or new_session_id()
    now = now_iso or _utc_now_iso()
    created = created_at or now
    if started_at is not None:
        started = started_at
    else:
        started = now if status == "running" else None

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "INSERT INTO trading_sessions "
            "(session_id, created_at, started_at, ended_at, status, mode, "
            " trigger_source, config_json, executor_version, replay_start, "
            " replay_end, harness_db_path, pnl_rollup_json, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (sid, created, started, None, status, mode, trigger_source,
             config_json, executor_version, replay_start, replay_end,
             harness_db_path, None, notes),
        )
        conn.commit()
    finally:
        conn.close()
    return sid


def update_session(db_path: Path | str = DB_PATH, *, session_id: str,
                   **fields: Any) -> None:
    """Set whitelisted columns on a session row. Raises if no such row."""
    bad = set(fields) - _UPDATABLE
    if bad:
        raise ValueError(f"cannot update non-whitelisted columns: {sorted(bad)}")
    if not fields:
        return
    assignments = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [session_id]
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            f"UPDATE trading_sessions SET {assignments} WHERE session_id = ?",
            values,
        )
        if cur.rowcount == 0:
            raise ValueError(
                f"no trading_sessions row for session_id={session_id!r}")
        conn.commit()
    finally:
        conn.close()


def close_session(db_path: Path | str = DB_PATH, session_id: str = "", *,
                  status: str = "completed",
                  pnl_rollup_json: str | None = None,
                  ended_at: str | None = None,
                  now_iso: str | None = None) -> None:
    """Mark a session ended: set status + ended_at (+ optional P&L rollup)."""
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid status {status!r}; "
                         f"expected one of {VALID_STATUSES}")
    fields: dict[str, Any] = {
        "status": status,
        "ended_at": ended_at or now_iso or _utc_now_iso(),
    }
    if pnl_rollup_json is not None:
        fields["pnl_rollup_json"] = pnl_rollup_json
    update_session(db_path, session_id=session_id, **fields)


def get_session(db_path: Path | str = DB_PATH, *,
                session_id: str) -> dict | None:
    """Return the session row as a dict, or None if absent."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM trading_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_sessions(db_path: Path | str = DB_PATH, *,
                  limit: int = 200) -> list[dict]:
    """Return session rows, newest first (history / analyze index)."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM trading_sessions "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def mark_interrupted_running_sessions(db_path: Path | str = DB_PATH, *,
                                      trigger_source: str = "gui",
                                      now_iso: str | None = None) -> int:
    """On backend startup, any session still 'running' was orphaned when the
    prior process died (the in-memory task registry did not survive). Mark
    them 'interrupted' rather than trusting a dead process is alive
    (freshness-as-verification; the long-lived-collector lesson). No
    auto-resume this cycle. Returns the number of rows updated.

    SCOPED to trigger_source (default 'gui'): a caller must sweep only sessions
    it owns. The scheduled task also creates 'running' rows (for the ~1s its
    process is mid-run); the GUI backend did not spawn those, cannot know their
    state, and must NOT mark them interrupted -- that would be a wrong-owner
    action and a race against a live scheduled fire. Cleanup of crashed
    *scheduled* sessions is a separate concern owned by the scheduled path.
    """
    now = now_iso or _utc_now_iso()
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            "UPDATE trading_sessions SET status = 'interrupted', ended_at = ? "
            "WHERE status = 'running' AND trigger_source = ?",
            (now, trigger_source),
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()
