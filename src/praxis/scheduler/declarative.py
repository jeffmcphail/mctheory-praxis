"""
Declarative Scheduling & Tick DAG Engine (Phase 3.13/3.14, §19.4-19.8).

Phase 3.13: Models declare data/signal/execution requirements in YAML.
            Scheduler infers explicit schedules from these declarations.

Phase 3.14: Each scheduler tick produces a DAG of primitive operations.
            Shared data loads are merged across models via needs filter.
            DAG executes in topological order with resource limits.

Usage:
    # Declarative: infer schedules from model config
    schedules = infer_schedules(model_config)

    # DAG: build and execute
    dag = TickDAG()
    dag.add_node(DAGNode("load_gld", "data_load", ...))
    dag.add_node(DAGNode("compute_signal", "compute", ...))
    dag.add_edge("load_gld", "compute_signal")
    results = dag.execute(executor_fn)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from praxis.config import ModelConfig
from praxis.logger.core import PraxisLogger
from praxis.scheduler import Schedule


# ═══════════════════════════════════════════════════════════════════
#  Phase 3.13: Declarative Scheduling (§19.8)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SchedulingDeclaration:
    """
    Parsed from model.scheduling section of YAML config.

    Represents what a model declares about its operational needs.
    """
    state: str = "active"
    priority: int = 50
    data_frequency: str = "daily"
    data_lookback: int = 252
    data_sources: list[str] = field(default_factory=lambda: ["yfinance"])
    signal_schedule: str = "market_open"  # "market_open", "market_close", cron
    execution_mode: str = "paper"
    retrain_frequency: str | None = None  # "daily", "weekly", "monthly"
    dependencies: list[str] = field(default_factory=list)


def parse_scheduling_declaration(config: ModelConfig) -> SchedulingDeclaration:
    """
    Extract scheduling declaration from model config.

    The scheduling section is stored in config.workflow or as
    top-level scheduling key. Falls back to sensible defaults.
    """
    decl = SchedulingDeclaration()

    # Check workflow section
    wf = config.workflow
    if wf:
        wf_dict = wf.model_dump() if hasattr(wf, "model_dump") else {}
        scheduling = wf_dict.get("scheduling", {})
    else:
        scheduling = {}

    if scheduling:
        decl.state = scheduling.get("state", decl.state)
        decl.priority = scheduling.get("priority", decl.priority)

        data = scheduling.get("data", {})
        if data:
            decl.data_frequency = data.get("frequency", decl.data_frequency)
            decl.data_lookback = data.get("lookback_periods", decl.data_lookback)
            decl.data_sources = data.get("sources", decl.data_sources)

        signals = scheduling.get("signals", {})
        if signals:
            decl.signal_schedule = signals.get("schedule", decl.signal_schedule)

        execution = scheduling.get("execution", {})
        if execution:
            decl.execution_mode = execution.get("mode", decl.execution_mode)

        retrain = scheduling.get("cpo_retrain", {})
        if retrain:
            decl.retrain_frequency = retrain.get("frequency")

        deps = scheduling.get("dependencies", {})
        if isinstance(deps, dict):
            for dep_list in deps.values():
                if isinstance(dep_list, list):
                    decl.dependencies.extend(dep_list)
        elif isinstance(deps, list):
            decl.dependencies = deps

    return decl


# Schedule templates for common patterns
_SCHEDULE_TEMPLATES = {
    "market_open": "30 9 * * 1-5",    # 09:30 ET weekdays
    "market_close": "0 16 * * 1-5",    # 16:00 ET weekdays
    "pre_market": "0 6 * * 1-5",       # 06:00 ET weekdays
    "daily": "0 18 * * 1-5",           # 18:00 ET weekdays (after close)
    "weekly": "0 18 * * 5",            # Friday 18:00 ET
    "monthly": "0 18 1 * *",           # 1st of month 18:00 ET
}


def infer_schedules(config: ModelConfig) -> list[Schedule]:
    """
    §19.8: Infer explicit Schedule objects from model declarations.

    Given a model config with scheduling declarations, produce the
    concrete cron schedules the scheduler should run.

    Returns list of Schedule objects ready for PraxisScheduler.add_schedule().
    """
    decl = parse_scheduling_declaration(config)
    model_name = config.model.name
    schedules = []
    log = PraxisLogger.instance()

    if decl.state not in ("active", "paper", "scheduled"):
        log.debug(
            f"Model '{model_name}' state={decl.state}, no schedules inferred",
            tags={"scheduler", "declarative"},
        )
        return schedules

    # 1. Data load schedule (pre-market)
    data_cron = _SCHEDULE_TEMPLATES.get("pre_market", "0 6 * * 1-5")
    schedules.append(Schedule(
        name=f"{model_name}__data_load",
        cron=data_cron,
        action="data_load",
        params={
            "model_name": model_name,
            "sources": decl.data_sources,
            "lookback": decl.data_lookback,
            "frequency": decl.data_frequency,
        },
        enabled=True,
        priority=decl.priority,
    ))

    # 2. Signal computation schedule
    signal_cron = _SCHEDULE_TEMPLATES.get(
        decl.signal_schedule,
        decl.signal_schedule,  # Allow raw cron expression
    )
    schedules.append(Schedule(
        name=f"{model_name}__compute_signal",
        cron=signal_cron,
        action="compute_signal",
        params={
            "model_name": model_name,
        },
        enabled=True,
        priority=decl.priority,
    ))

    # 3. Execution schedule (same as signal by default)
    if decl.execution_mode in ("paper", "live"):
        schedules.append(Schedule(
            name=f"{model_name}__execute",
            cron=signal_cron,
            action="execute",
            params={
                "model_name": model_name,
                "mode": decl.execution_mode,
            },
            enabled=True,
            priority=decl.priority + 1,  # Execute after signal
        ))

    # 4. Retrain schedule (CPO)
    if decl.retrain_frequency:
        retrain_cron = _SCHEDULE_TEMPLATES.get(
            decl.retrain_frequency,
            decl.retrain_frequency,
        )
        schedules.append(Schedule(
            name=f"{model_name}__retrain",
            cron=retrain_cron,
            action="retrain",
            params={
                "model_name": model_name,
            },
            enabled=True,
            priority=decl.priority + 10,  # Lower priority than execution
        ))

    log.info(
        f"Inferred {len(schedules)} schedules for '{model_name}' "
        f"(state={decl.state}, signal={decl.signal_schedule})",
        tags={"scheduler", "declarative"},
    )

    return schedules


# ═══════════════════════════════════════════════════════════════════
#  Phase 3.14: Tick DAG Engine (§19.4-19.5)
# ═══════════════════════════════════════════════════════════════════

class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class DAGNode:
    """A single operation in the tick DAG."""
    id: str
    node_type: str  # "data_load", "validate", "compute", "execute"
    model_name: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    status: NodeStatus = NodeStatus.PENDING
    duration_seconds: float = 0.0
    error: str | None = None
    result: Any = None

    @property
    def is_complete(self) -> bool:
        return self.status in (NodeStatus.COMPLETED, NodeStatus.FAILED, NodeStatus.SKIPPED)


@dataclass
class DAGEdge:
    """Dependency: from_id must complete before to_id starts."""
    from_id: str
    to_id: str


@dataclass
class DAGExecutionResult:
    """Result of executing the full DAG."""
    nodes: list[DAGNode]
    total_duration: float = 0.0
    completed: int = 0
    failed: int = 0
    skipped: int = 0

    @property
    def all_succeeded(self) -> bool:
        return self.failed == 0


class TickDAG:
    """
    §19.5: Directed Acyclic Graph of operations for one scheduler tick.

    Nodes are primitive operations (data_load, validate, compute, execute).
    Edges represent dependencies. Execution respects topological order
    and resource limits.
    """

    def __init__(self):
        self._nodes: dict[str, DAGNode] = {}
        self._edges: list[DAGEdge] = []
        self._log = PraxisLogger.instance()

    def add_node(self, node: DAGNode) -> None:
        if node.id in self._nodes:
            raise ValueError(f"Duplicate node id: {node.id}")
        self._nodes[node.id] = node

    def add_edge(self, from_id: str, to_id: str) -> None:
        if from_id not in self._nodes:
            raise ValueError(f"Unknown source node: {from_id}")
        if to_id not in self._nodes:
            raise ValueError(f"Unknown target node: {to_id}")
        self._edges.append(DAGEdge(from_id=from_id, to_id=to_id))

    @property
    def nodes(self) -> list[DAGNode]:
        return list(self._nodes.values())

    @property
    def edges(self) -> list[DAGEdge]:
        return list(self._edges)

    @property
    def size(self) -> int:
        return len(self._nodes)

    def topological_order(self) -> list[str]:
        """
        Kahn's algorithm for topological sort.

        Returns node IDs in execution order.
        Raises ValueError on cycles.
        """
        in_degree: dict[str, int] = {nid: 0 for nid in self._nodes}
        adjacency: dict[str, list[str]] = {nid: [] for nid in self._nodes}

        for edge in self._edges:
            adjacency[edge.from_id].append(edge.to_id)
            in_degree[edge.to_id] += 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        result = []

        while queue:
            # Sort for deterministic ordering within same level
            queue.sort()
            nid = queue.pop(0)
            result.append(nid)

            for neighbor in adjacency[nid]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(self._nodes):
            raise ValueError("Cycle detected in DAG")

        return result

    def get_dependencies(self, node_id: str) -> list[str]:
        """Get all nodes that must complete before node_id."""
        return [e.from_id for e in self._edges if e.to_id == node_id]

    def get_dependents(self, node_id: str) -> list[str]:
        """Get all nodes that depend on node_id."""
        return [e.to_id for e in self._edges if e.from_id == node_id]

    def execute(
        self,
        executor: Callable[[DAGNode], Any] | None = None,
        max_failures: int = 3,
    ) -> DAGExecutionResult:
        """
        Execute DAG in topological order.

        Args:
            executor: Function to call for each node. Receives DAGNode,
                      should return result or raise on failure.
                      Default: no-op (marks all as completed).
            max_failures: Stop after this many failures.

        Returns:
            DAGExecutionResult with per-node status.
        """
        import time

        order = self.topological_order()
        start_time = time.monotonic()
        failures = 0

        self._log.info(
            f"DAG execution: {len(order)} nodes",
            tags={"scheduler", "dag"},
        )

        for nid in order:
            node = self._nodes[nid]

            # Check dependencies
            deps = self.get_dependencies(nid)
            dep_failed = any(
                self._nodes[d].status == NodeStatus.FAILED for d in deps
            )
            if dep_failed:
                node.status = NodeStatus.SKIPPED
                node.error = "Dependency failed"
                self._log.debug(
                    f"DAG: skipping {nid} (dependency failed)",
                    tags={"scheduler", "dag"},
                )
                continue

            if failures >= max_failures:
                node.status = NodeStatus.SKIPPED
                node.error = "Max failures reached"
                continue

            # Execute
            node.status = NodeStatus.RUNNING
            node_start = time.monotonic()

            try:
                if executor:
                    node.result = executor(node)
                node.status = NodeStatus.COMPLETED
            except Exception as e:
                node.status = NodeStatus.FAILED
                node.error = str(e)
                failures += 1
                self._log.error(
                    f"DAG: node {nid} failed: {e}",
                    tags={"scheduler", "dag"},
                )

            node.duration_seconds = time.monotonic() - node_start

        total_duration = time.monotonic() - start_time

        result = DAGExecutionResult(
            nodes=list(self._nodes.values()),
            total_duration=total_duration,
            completed=sum(1 for n in self._nodes.values() if n.status == NodeStatus.COMPLETED),
            failed=sum(1 for n in self._nodes.values() if n.status == NodeStatus.FAILED),
            skipped=sum(1 for n in self._nodes.values() if n.status == NodeStatus.SKIPPED),
        )

        self._log.info(
            f"DAG complete: {result.completed} ok, {result.failed} failed, "
            f"{result.skipped} skipped, {total_duration:.3f}s",
            tags={"scheduler", "dag"},
        )

        return result


def build_tick_dag(
    models: list[dict[str, Any]],
    merge_data_loads: bool = True,
) -> TickDAG:
    """
    §19.5: Build a tick DAG from model requirements.

    Each model needs: data_load → validate → compute → execute.
    If merge_data_loads is True, shared data needs are deduplicated
    (e.g., two models needing SP500 equity → one merged load).

    Args:
        models: List of model requirement dicts with keys:
            name, tickers, source, priority
        merge_data_loads: Merge overlapping data loads.

    Returns:
        TickDAG ready for execution.
    """
    dag = TickDAG()

    # Track data loads for merging
    data_load_map: dict[str, str] = {}  # (source, ticker_key) → node_id

    for m in models:
        name = m["name"]
        tickers = m.get("tickers", [])
        source = m.get("source", "yfinance")
        ticker_key = ",".join(sorted(tickers))

        # ── Data load (possibly merged) ───────────────────────
        load_key = f"{source}:{ticker_key}"
        if merge_data_loads and load_key in data_load_map:
            load_node_id = data_load_map[load_key]
        else:
            load_node_id = f"load_{name}"
            dag.add_node(DAGNode(
                id=load_node_id,
                node_type="data_load",
                model_name=name,
                params={"tickers": tickers, "source": source},
            ))
            if merge_data_loads:
                data_load_map[load_key] = load_node_id

        # ── Validate ──────────────────────────────────────────
        validate_id = f"validate_{name}"
        dag.add_node(DAGNode(
            id=validate_id,
            node_type="validate",
            model_name=name,
        ))
        dag.add_edge(load_node_id, validate_id)

        # ── Compute ───────────────────────────────────────────
        compute_id = f"compute_{name}"
        dag.add_node(DAGNode(
            id=compute_id,
            node_type="compute",
            model_name=name,
        ))
        dag.add_edge(validate_id, compute_id)

        # ── Execute ───────────────────────────────────────────
        execute_id = f"execute_{name}"
        dag.add_node(DAGNode(
            id=execute_id,
            node_type="execute",
            model_name=name,
        ))
        dag.add_edge(compute_id, execute_id)

    return dag
