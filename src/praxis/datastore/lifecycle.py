"""
Model Lifecycle State Machine (Phase 2.12, §19.2).

States: DRAFT → ACTIVE → PAUSED → RETIRED
         DRAFT → BACKTESTING → DRAFT
         ACTIVE → ERROR → ACTIVE (after fix)

Each transition creates a new temporal version in dim_model_state.

Usage:
    lifecycle = ModelLifecycle(conn)
    lifecycle.create("my_model_bpk", execution_mode="backtest_only")
    lifecycle.activate("my_model_bpk", reason="passed validation")
    lifecycle.pause("my_model_bpk", reason="reviewing performance")
    lifecycle.retire("my_model_bpk", reason="replaced by v2")
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Optional

import duckdb

from praxis.datastore.keys import EntityKeys
from praxis.logger.core import PraxisLogger


# ═══════════════════════════════════════════════════════════════════
#  Valid Transitions (§19.2 state machine)
# ═══════════════════════════════════════════════════════════════════

VALID_TRANSITIONS: dict[str, set[str]] = {
    "DRAFT":       {"ACTIVE", "BACKTESTING", "RETIRED", "SCHEDULED"},
    "ACTIVE":      {"PAUSED", "RETIRED", "ERROR"},
    "PAUSED":      {"ACTIVE", "RETIRED"},
    "QUEUED":      {"ACTIVE", "RETIRED"},
    "BACKTESTING": {"DRAFT", "ERROR"},
    "SCHEDULED":   {"ACTIVE", "DRAFT", "RETIRED"},
    "ERROR":       {"ACTIVE", "PAUSED", "RETIRED", "DRAFT"},
    "RETIRED":     set(),  # Terminal state
}

ALL_STATES = set(VALID_TRANSITIONS.keys())


class ModelLifecycle:
    """§19.2: Model lifecycle state machine with temporal audit trail."""

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self._conn = conn
        self._log = PraxisLogger.instance()

    # ── State transitions ─────────────────────────────────────

    def create(
        self,
        model_bpk: str,
        *,
        execution_mode: str = "backtest_only",
        data_frequency: str = "daily",
        data_lookback_periods: int = 252,
        signal_schedule: str | None = None,
        priority: int = 50,
        changed_by: str = "user",
    ) -> int:
        """Create a new model in DRAFT state. Returns model_state_base_id."""
        keys = EntityKeys.create(model_bpk)

        self._conn.execute("""
            INSERT INTO dim_model_state (
                model_state_hist_id, model_state_base_id, model_state_bpk,
                state, priority, data_frequency, data_lookback_periods,
                signal_schedule, execution_mode,
                changed_by, change_reason
            ) VALUES ($1, $2, $3, 'DRAFT', $4, $5, $6, $7, $8, $9, 'Model created')
        """, [
            keys.hist_id, keys.base_id, keys.bpk,
            priority, data_frequency, data_lookback_periods,
            signal_schedule, execution_mode,
            changed_by,
        ])

        self._log.info(
            f"Model lifecycle: created {model_bpk} in DRAFT",
            tags={"model_lifecycle"},
        )
        return keys.base_id

    def transition(
        self,
        model_bpk: str,
        new_state: str,
        *,
        reason: str = "",
        changed_by: str = "user",
        **kwargs,
    ) -> bool:
        """
        Generic state transition with validation.

        Returns True if transition succeeded, False if invalid.
        """
        new_state = new_state.upper()
        if new_state not in ALL_STATES:
            raise ValueError(f"Invalid state: {new_state}. Valid: {sorted(ALL_STATES)}")

        current = self.get_state(model_bpk)
        if current is None:
            raise ValueError(f"Model {model_bpk} not found in dim_model_state")

        current_state = current["state"]
        valid = VALID_TRANSITIONS.get(current_state, set())

        if new_state not in valid:
            self._log.warning(
                f"Invalid transition: {current_state} → {new_state} "
                f"for {model_bpk}. Valid: {sorted(valid)}",
                tags={"model_lifecycle"},
            )
            return False

        # Create new version with updated state
        new_keys = EntityKeys.new_version(
            current["model_state_bpk"],
            current["model_state_base_id"],
        )

        self._conn.execute("""
            INSERT INTO dim_model_state (
                model_state_hist_id, model_state_base_id, model_state_bpk,
                state, operational_start, operational_end, priority,
                data_frequency, data_lookback_periods, signal_schedule,
                execution_mode, estimated_runtime_seconds, memory_requirement_mb,
                gpu_required, changed_by, change_reason
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16
            )
        """, [
            new_keys.hist_id, new_keys.base_id, new_keys.bpk,
            new_state,
            kwargs.get("operational_start", current.get("operational_start")),
            kwargs.get("operational_end", current.get("operational_end")),
            kwargs.get("priority", current.get("priority")),
            kwargs.get("data_frequency", current.get("data_frequency")),
            kwargs.get("data_lookback_periods", current.get("data_lookback_periods")),
            kwargs.get("signal_schedule", current.get("signal_schedule")),
            kwargs.get("execution_mode", current.get("execution_mode")),
            kwargs.get("estimated_runtime_seconds", current.get("estimated_runtime_seconds")),
            kwargs.get("memory_requirement_mb", current.get("memory_requirement_mb")),
            kwargs.get("gpu_required", current.get("gpu_required")),
            changed_by,
            reason,
        ])

        self._log.info(
            f"Model lifecycle: {model_bpk} {current_state} → {new_state}"
            + (f" ({reason})" if reason else ""),
            tags={"model_lifecycle"},
        )
        return True

    # ── Convenience methods ───────────────────────────────────

    def activate(self, model_bpk: str, *, reason: str = "", changed_by: str = "user", **kw) -> bool:
        return self.transition(model_bpk, "ACTIVE", reason=reason, changed_by=changed_by, **kw)

    def pause(self, model_bpk: str, *, reason: str = "", changed_by: str = "user") -> bool:
        return self.transition(model_bpk, "PAUSED", reason=reason, changed_by=changed_by)

    def retire(self, model_bpk: str, *, reason: str = "", changed_by: str = "user") -> bool:
        return self.transition(model_bpk, "RETIRED", reason=reason, changed_by=changed_by)

    def start_backtest(self, model_bpk: str, *, changed_by: str = "system") -> bool:
        return self.transition(model_bpk, "BACKTESTING", reason="backtest started", changed_by=changed_by)

    def complete_backtest(self, model_bpk: str, *, changed_by: str = "system") -> bool:
        return self.transition(model_bpk, "DRAFT", reason="backtest completed", changed_by=changed_by)

    def mark_error(self, model_bpk: str, *, reason: str = "", changed_by: str = "scheduler") -> bool:
        return self.transition(model_bpk, "ERROR", reason=reason, changed_by=changed_by)

    # ── Queries ───────────────────────────────────────────────

    def get_state(self, model_bpk: str) -> dict[str, Any] | None:
        """Get current state of a model (latest version)."""
        keys = EntityKeys.create(model_bpk)
        row = self._conn.execute("""
            SELECT * EXCLUDE (rn) FROM (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY model_state_base_id
                           ORDER BY model_state_hist_id DESC
                       ) AS rn
                FROM dim_model_state
                WHERE model_state_base_id = $1
            ) WHERE rn = 1
        """, [keys.base_id]).fetchone()

        if row is None:
            return None
        cols = [desc[0] for desc in self._conn.description]
        return dict(zip(cols, row))

    def get_history(self, model_bpk: str) -> list[dict[str, Any]]:
        """Get full state history of a model (audit trail)."""
        keys = EntityKeys.create(model_bpk)
        rows = self._conn.execute("""
            SELECT model_state_hist_id, state, changed_by, change_reason
            FROM dim_model_state
            WHERE model_state_base_id = $1
            ORDER BY model_state_hist_id ASC
        """, [keys.base_id]).fetchall()

        return [
            {"timestamp": r[0], "state": r[1], "changed_by": r[2], "reason": r[3]}
            for r in rows
        ]

    def get_active_models(self) -> list[dict[str, Any]]:
        """Get all models in ACTIVE or QUEUED state."""
        rows = self._conn.execute("""
            SELECT * EXCLUDE (rn) FROM (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY model_state_base_id
                           ORDER BY model_state_hist_id DESC
                       ) AS rn
                FROM dim_model_state
            ) WHERE rn = 1 AND state IN ('ACTIVE', 'QUEUED')
        """).fetchall()

        cols = [desc[0] for desc in self._conn.description]
        return [dict(zip(cols, r)) for r in rows]

    def count(self, state: str | None = None) -> int:
        """Count models, optionally filtered by state."""
        if state:
            return self._conn.execute("""
                SELECT COUNT(DISTINCT model_state_base_id) FROM (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY model_state_base_id
                        ORDER BY model_state_hist_id DESC
                    ) AS rn FROM dim_model_state
                ) WHERE rn = 1 AND state = $1
            """, [state.upper()]).fetchone()[0]
        else:
            return self._conn.execute(
                "SELECT COUNT(DISTINCT model_state_base_id) FROM dim_model_state"
            ).fetchone()[0]
