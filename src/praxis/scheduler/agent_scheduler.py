"""
Agent-Driven Autonomous Scheduler (Phase 4.15 + 4.17, §19.10).

Extends PraxisScheduler with:
- Health monitoring: watch for repeated failures, auto-retry
- State transitions: DRAFT → BACKTESTING → PAPER → ACTIVE lifecycle
- Priority optimization: adjust based on model health/performance
- Universe merging: identify overlapping data needs, deduplicate loads
- Holistic scheduling: merged universes, cross-model optimization

Usage:
    agent_scheduler = AgentScheduler(scheduler, config)
    agent_scheduler.monitor_health()
    agent_scheduler.optimize_priorities()
    merged = agent_scheduler.merge_universes(model_configs)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from praxis.logger.core import PraxisLogger


# ═══════════════════════════════════════════════════════════════════
#  Model Lifecycle (§19.2)
# ═══════════════════════════════════════════════════════════════════

class ModelState(str, Enum):
    DRAFT = "draft"
    SCHEDULED = "scheduled"
    QUEUED = "queued"
    ACTIVE = "active"
    PAUSED = "paused"
    BACKTESTING = "backtesting"
    ERROR = "error"
    RETIRED = "retired"


# Valid state transitions
_TRANSITIONS: dict[ModelState, set[ModelState]] = {
    ModelState.DRAFT: {ModelState.ACTIVE, ModelState.BACKTESTING, ModelState.SCHEDULED},
    ModelState.SCHEDULED: {ModelState.ACTIVE, ModelState.QUEUED, ModelState.RETIRED},
    ModelState.QUEUED: {ModelState.ACTIVE, ModelState.RETIRED},
    ModelState.ACTIVE: {ModelState.PAUSED, ModelState.ERROR, ModelState.RETIRED},
    ModelState.PAUSED: {ModelState.ACTIVE, ModelState.RETIRED},
    ModelState.BACKTESTING: {ModelState.DRAFT, ModelState.ACTIVE, ModelState.ERROR},
    ModelState.ERROR: {ModelState.ACTIVE, ModelState.PAUSED, ModelState.RETIRED},
    ModelState.RETIRED: set(),  # Terminal
}


@dataclass
class ModelStateRecord:
    """Track a model's lifecycle state and operational params."""
    model_id: str
    state: ModelState = ModelState.DRAFT
    priority: int = 50
    data_frequency: str = "daily"
    signal_schedule: str = "market_open"
    execution_mode: str = "backtest_only"
    depends_on_models: list[str] = field(default_factory=list)
    depends_on_data: list[str] = field(default_factory=list)
    universe: list[str] = field(default_factory=list)
    failure_count: int = 0
    last_success: datetime | None = None
    last_failure: datetime | None = None
    changed_by: str = "user"
    change_reason: str = ""


def can_transition(from_state: ModelState, to_state: ModelState) -> bool:
    """Check if a state transition is valid."""
    return to_state in _TRANSITIONS.get(from_state, set())


def transition(record: ModelStateRecord, to_state: ModelState, by: str = "agent", reason: str = "") -> bool:
    """Attempt a state transition. Returns True if successful."""
    if not can_transition(record.state, to_state):
        return False
    record.state = to_state
    record.changed_by = by
    record.change_reason = reason
    return True


# ═══════════════════════════════════════════════════════════════════
#  Health Monitor
# ═══════════════════════════════════════════════════════════════════

@dataclass
class HealthStatus:
    """Health assessment for a model."""
    model_id: str
    is_healthy: bool = True
    failure_count: int = 0
    consecutive_failures: int = 0
    days_since_success: float = 0.0
    days_inactive: float = 0.0
    recommendation: str = ""  # "ok", "retry", "pause", "investigate", "retire"
    details: list[str] = field(default_factory=list)


@dataclass
class AgentSchedulerConfig:
    """Configuration for the agent-driven scheduler."""
    # Health thresholds
    max_consecutive_failures: int = 3
    max_total_failures: int = 10
    stale_days_threshold: float = 30.0
    retry_delay_seconds: float = 60.0

    # Priority
    min_priority: int = 1
    max_priority: int = 100
    priority_boost_on_success: int = 5
    priority_penalty_on_failure: int = 10

    # Permissions
    can_change_priority: bool = True
    can_pause_models: bool = True
    can_activate_models: bool = False   # Requires human approval
    can_retry_failed: bool = True
    can_merge_universes: bool = True


class HealthMonitor:
    """
    Monitors model health and recommends actions (§19.10.1).
    """

    def __init__(self, config: AgentSchedulerConfig | None = None):
        self._config = config or AgentSchedulerConfig()

    def assess(self, record: ModelStateRecord, now: datetime | None = None) -> HealthStatus:
        """Assess health of a single model."""
        now = now or datetime.now(timezone.utc)
        status = HealthStatus(
            model_id=record.model_id,
            failure_count=record.failure_count,
        )

        # Check consecutive failures
        if record.failure_count >= self._config.max_consecutive_failures:
            status.is_healthy = False
            status.consecutive_failures = record.failure_count
            status.recommendation = "pause"
            status.details.append(
                f"{record.failure_count} failures >= threshold {self._config.max_consecutive_failures}"
            )

        # Check staleness
        if record.last_success is not None:
            delta = (now - record.last_success).total_seconds() / 86400
            status.days_since_success = delta
            if delta > self._config.stale_days_threshold:
                status.recommendation = "investigate"
                status.details.append(
                    f"No success in {delta:.1f} days"
                )

        # Check for ERROR state
        if record.state == ModelState.ERROR:
            status.is_healthy = False
            if record.failure_count < self._config.max_total_failures:
                status.recommendation = "retry"
            else:
                status.recommendation = "retire"
            status.details.append(f"Model in ERROR state")

        if not status.details:
            status.recommendation = "ok"

        return status

    def assess_all(self, records: list[ModelStateRecord]) -> list[HealthStatus]:
        """Assess health of all models."""
        return [self.assess(r) for r in records]


# ═══════════════════════════════════════════════════════════════════
#  Priority Optimizer
# ═══════════════════════════════════════════════════════════════════

class PriorityOptimizer:
    """
    Dynamically adjusts model priorities based on performance (§19.10.2).
    """

    def __init__(self, config: AgentSchedulerConfig | None = None):
        self._config = config or AgentSchedulerConfig()

    def adjust_on_success(self, record: ModelStateRecord) -> int:
        """Boost priority (lower number = higher priority) on success."""
        if not self._config.can_change_priority:
            return record.priority
        new_priority = max(
            self._config.min_priority,
            record.priority - self._config.priority_boost_on_success,
        )
        record.priority = new_priority
        record.failure_count = 0
        record.last_success = datetime.now(timezone.utc)
        return new_priority

    def adjust_on_failure(self, record: ModelStateRecord) -> int:
        """Penalize priority on failure."""
        if not self._config.can_change_priority:
            return record.priority
        new_priority = min(
            self._config.max_priority,
            record.priority + self._config.priority_penalty_on_failure,
        )
        record.priority = new_priority
        record.failure_count += 1
        record.last_failure = datetime.now(timezone.utc)
        return new_priority

    def rebalance(self, records: list[ModelStateRecord]) -> list[ModelStateRecord]:
        """
        Rebalance priorities across all models.

        Active models get boosted, error/paused get penalized,
        ensuring healthy models run first.
        """
        for record in records:
            if record.state == ModelState.ACTIVE and record.failure_count == 0:
                record.priority = max(self._config.min_priority, record.priority - 2)
            elif record.state in (ModelState.ERROR, ModelState.PAUSED):
                record.priority = min(self._config.max_priority, record.priority + 5)

        return sorted(records, key=lambda r: r.priority)


# ═══════════════════════════════════════════════════════════════════
#  Universe Merger (§19.3 Holistic)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class MergedUniverse:
    """Result of merging data needs across models."""
    assets: set[str] = field(default_factory=set)
    models_served: list[str] = field(default_factory=list)
    data_frequency: str = "daily"
    max_lookback: int = 0
    sources: list[str] = field(default_factory=list)

    @property
    def n_assets(self) -> int:
        return len(self.assets)

    @property
    def n_models(self) -> int:
        return len(self.models_served)


@dataclass
class UniverseMergeResult:
    """Full result of universe merge analysis."""
    merged_universes: list[MergedUniverse] = field(default_factory=list)
    total_assets_before: int = 0
    total_assets_after: int = 0
    dedup_savings_pct: float = 0.0
    models_analyzed: int = 0


class UniverseMerger:
    """
    Merge overlapping data needs across models (§19.3 holistic).

    Instead of each model loading its own universe independently,
    identify overlapping securities and merge into consolidated loads.
    """

    def analyze(self, records: list[ModelStateRecord]) -> UniverseMergeResult:
        """
        Analyze model universes for merge opportunities.

        Groups models by data_frequency, then merges overlapping universes.
        """
        active = [r for r in records if r.state in (ModelState.ACTIVE, ModelState.QUEUED)]
        result = UniverseMergeResult(models_analyzed=len(active))

        if not active:
            return result

        # Group by frequency
        by_freq: dict[str, list[ModelStateRecord]] = {}
        for r in active:
            by_freq.setdefault(r.data_frequency, []).append(r)

        total_before = 0
        total_after = 0

        for freq, models in by_freq.items():
            # Merge all universes for this frequency
            all_assets: set[str] = set()
            model_ids = []
            sources: set[str] = set()

            for m in models:
                total_before += len(m.universe)
                all_assets.update(m.universe)
                model_ids.append(m.model_id)
                sources.update(m.depends_on_data)

            total_after += len(all_assets)

            merged = MergedUniverse(
                assets=all_assets,
                models_served=model_ids,
                data_frequency=freq,
                sources=sorted(sources),
            )
            result.merged_universes.append(merged)

        result.total_assets_before = total_before
        result.total_assets_after = total_after
        if total_before > 0:
            result.dedup_savings_pct = 1.0 - total_after / total_before

        return result


# ═══════════════════════════════════════════════════════════════════
#  Agent Scheduler
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AgentAction:
    """An action taken by the agent scheduler."""
    action_type: str       # "retry", "pause", "priority_change", "state_change", "universe_merge"
    model_id: str = ""
    details: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    requires_approval: bool = False


class AgentScheduler:
    """
    Agent-driven autonomous scheduler (§19.10).

    Wraps the base scheduler with intelligent health monitoring,
    automatic state transitions, priority optimization, and
    holistic universe merging.
    """

    def __init__(self, config: AgentSchedulerConfig | None = None):
        self._config = config or AgentSchedulerConfig()
        self._log = PraxisLogger.instance()
        self._health = HealthMonitor(self._config)
        self._priority = PriorityOptimizer(self._config)
        self._merger = UniverseMerger()
        self._models: dict[str, ModelStateRecord] = {}
        self._action_log: list[AgentAction] = []

    @property
    def config(self) -> AgentSchedulerConfig:
        return self._config

    @property
    def models(self) -> dict[str, ModelStateRecord]:
        return dict(self._models)

    @property
    def action_log(self) -> list[AgentAction]:
        return list(self._action_log)

    def register_model(self, record: ModelStateRecord) -> None:
        """Register a model for agent management."""
        self._models[record.model_id] = record

    def get_model(self, model_id: str) -> ModelStateRecord | None:
        return self._models.get(model_id)

    # ── Health Monitoring ─────────────────────────────────────

    def monitor_health(self) -> list[AgentAction]:
        """
        Assess all models and take corrective actions.

        Returns list of actions taken.
        """
        actions = []
        records = list(self._models.values())
        statuses = self._health.assess_all(records)

        for status in statuses:
            record = self._models.get(status.model_id)
            if record is None:
                continue

            if status.recommendation == "retry" and self._config.can_retry_failed:
                if transition(record, ModelState.ACTIVE, by="agent", reason="auto-retry"):
                    record.failure_count = 0
                    action = AgentAction("retry", record.model_id, "Auto-retry after failure")
                    actions.append(action)
                    self._action_log.append(action)

            elif status.recommendation == "pause" and self._config.can_pause_models:
                if transition(record, ModelState.PAUSED, by="agent",
                              reason=f"{status.consecutive_failures} consecutive failures"):
                    action = AgentAction("pause", record.model_id,
                                         f"Paused: {status.consecutive_failures} failures")
                    actions.append(action)
                    self._action_log.append(action)

            elif status.recommendation == "retire":
                if transition(record, ModelState.RETIRED, by="agent",
                              reason=f"Exceeded {self._config.max_total_failures} total failures"):
                    action = AgentAction("state_change", record.model_id, "Retired due to excess failures")
                    actions.append(action)
                    self._action_log.append(action)

        return actions

    # ── Priority Management ───────────────────────────────────

    def record_success(self, model_id: str) -> None:
        """Record a successful execution for a model."""
        record = self._models.get(model_id)
        if record is None:
            return
        old_priority = record.priority
        new_priority = self._priority.adjust_on_success(record)
        if old_priority != new_priority:
            action = AgentAction("priority_change", model_id,
                                 f"Priority {old_priority} → {new_priority} (success)")
            self._action_log.append(action)

    def record_failure(self, model_id: str) -> None:
        """Record a failed execution for a model."""
        record = self._models.get(model_id)
        if record is None:
            return
        old_priority = record.priority
        new_priority = self._priority.adjust_on_failure(record)
        action = AgentAction("priority_change", model_id,
                             f"Priority {old_priority} → {new_priority} (failure)")
        self._action_log.append(action)

    def optimize_priorities(self) -> list[ModelStateRecord]:
        """Rebalance all model priorities."""
        records = list(self._models.values())
        return self._priority.rebalance(records)

    # ── Universe Merging ──────────────────────────────────────

    def merge_universes(self) -> UniverseMergeResult:
        """Analyze and merge overlapping data universes."""
        records = list(self._models.values())
        return self._merger.analyze(records)

    # ── State Transitions ─────────────────────────────────────

    def transition_model(
        self, model_id: str, to_state: ModelState,
        by: str = "user", reason: str = "",
    ) -> bool:
        """
        Transition a model's state with validation.

        Returns True if transition succeeded.
        """
        record = self._models.get(model_id)
        if record is None:
            return False

        # Check permissions
        if to_state == ModelState.ACTIVE and not self._config.can_activate_models and by == "agent":
            action = AgentAction("state_change", model_id,
                                 f"Activation blocked — requires human approval",
                                 requires_approval=True)
            self._action_log.append(action)
            return False

        if transition(record, to_state, by=by, reason=reason):
            action = AgentAction("state_change", model_id,
                                 f"State → {to_state.value} by {by}: {reason}")
            self._action_log.append(action)
            return True
        return False

    # ── Execution Order ───────────────────────────────────────

    def get_execution_order(self) -> list[str]:
        """
        Get models in execution order: priority then dependencies.

        Returns model_ids sorted by priority with dependency ordering.
        """
        active = [
            r for r in self._models.values()
            if r.state in (ModelState.ACTIVE, ModelState.QUEUED)
        ]

        # Sort by priority
        active.sort(key=lambda r: r.priority)

        # Topological sort respecting dependencies
        ordered = []
        visited: set[str] = set()

        def visit(model_id: str):
            if model_id in visited:
                return
            visited.add(model_id)
            record = self._models.get(model_id)
            if record:
                for dep in record.depends_on_models:
                    if dep in self._models:
                        visit(dep)
                ordered.append(model_id)

        for r in active:
            visit(r.model_id)

        return ordered

    # ── Summary ───────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """Get agent scheduler summary."""
        by_state: dict[str, int] = {}
        for r in self._models.values():
            by_state[r.state.value] = by_state.get(r.state.value, 0) + 1

        merge = self.merge_universes()

        return {
            "total_models": len(self._models),
            "by_state": by_state,
            "actions_taken": len(self._action_log),
            "universe_dedup_savings": f"{merge.dedup_savings_pct:.1%}",
            "merged_universes": len(merge.merged_universes),
        }
