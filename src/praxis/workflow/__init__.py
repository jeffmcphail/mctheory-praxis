"""
Workflow Executor (Phase 4.1, §10).

Full DAG support for complex multi-step models. Supports:
- Step ordering via depends_on
- Output passing between steps
- Conditional branching (condition → if_true / if_false)
- for_each parallel execution
- Function resolution from registry

Usage:
    executor = WorkflowExecutor(registry=reg)
    executor.add_step(WorkflowStep(id="load", function="load_data", params={...}))
    executor.add_step(WorkflowStep(id="test", function="run_adf", depends_on=["load"]))
    result = executor.run()
    print(result.outputs["test"])
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from praxis.logger.core import PraxisLogger


# ═══════════════════════════════════════════════════════════════════
#  Data Types
# ═══════════════════════════════════════════════════════════════════

class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """One step in a workflow DAG."""
    id: str
    function: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)

    # Conditional branching (§10)
    condition: str | None = None      # e.g. "step_a.output.count > 100"
    if_true: dict | None = None       # {function: ..., params: ...}
    if_false: dict | None = None      # {function: ..., params: ...}

    # for_each parallel (§10)
    for_each: str | None = None       # e.g. "filter.output.candidates"
    for_each_as: str = "item"         # Variable name in params
    parallel: bool = False

    # Runtime state
    status: StepStatus = StepStatus.PENDING
    output: Any = None
    error: str | None = None
    duration_seconds: float = 0.0


@dataclass
class WorkflowResult:
    """Result of running a complete workflow."""
    steps_total: int = 0
    steps_completed: int = 0
    steps_failed: int = 0
    steps_skipped: int = 0
    outputs: dict[str, Any] = field(default_factory=dict)
    step_results: list[WorkflowStep] = field(default_factory=list)
    total_duration: float = 0.0
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.steps_failed == 0 and self.error is None


# ═══════════════════════════════════════════════════════════════════
#  Function Registry Interface
# ═══════════════════════════════════════════════════════════════════

class WorkflowFunctionRegistry:
    """
    Resolves function names to callables for the workflow executor.

    Can wrap the platform's FunctionRegistry or be used standalone
    with directly registered callables.
    """

    def __init__(self):
        self._functions: dict[str, Callable] = {}

    def register(self, name: str, fn: Callable) -> None:
        self._functions[name] = fn

    def resolve(self, name: str) -> Callable | None:
        return self._functions.get(name)

    def has(self, name: str) -> bool:
        return name in self._functions

    @property
    def names(self) -> list[str]:
        return list(self._functions.keys())


# ═══════════════════════════════════════════════════════════════════
#  Output Resolution
# ═══════════════════════════════════════════════════════════════════

def _resolve_reference(ref: str, outputs: dict[str, Any]) -> Any:
    """
    Resolve a dotted reference like 'step_a.output.candidates'.

    Supports:
    - step_id.output → full output
    - step_id.output.field → dict/attr lookup
    - step_id.output.field.subfield → nested lookup
    """
    parts = ref.strip().split(".")
    if len(parts) < 2:
        return outputs.get(parts[0])

    step_id = parts[0]
    if step_id not in outputs:
        return None

    value = outputs[step_id]
    for part in parts[1:]:
        if part == "output":
            continue  # Skip the literal "output" token
        if isinstance(value, dict):
            value = value.get(part)
        elif hasattr(value, part):
            value = getattr(value, part)
        else:
            return None

    return value


def _evaluate_condition(condition: str, outputs: dict[str, Any]) -> bool:
    """
    Evaluate a simple condition string.

    Supports: >, <, >=, <=, ==, !=
    Left side is a reference, right side is a literal.
    e.g. "step_a.output.count > 100"
    """
    for op in [">=", "<=", "!=", "==", ">", "<"]:
        if op in condition:
            left, right = condition.split(op, 1)
            left_val = _resolve_reference(left.strip(), outputs)
            right_val = right.strip()

            # Try numeric comparison
            try:
                right_val = float(right_val)
                left_val = float(left_val) if left_val is not None else 0
            except (ValueError, TypeError):
                # String comparison
                right_val = right_val.strip("'\"")
                left_val = str(left_val) if left_val is not None else ""

            if op == ">":
                return left_val > right_val
            elif op == "<":
                return left_val < right_val
            elif op == ">=":
                return left_val >= right_val
            elif op == "<=":
                return left_val <= right_val
            elif op == "==":
                return left_val == right_val
            elif op == "!=":
                return left_val != right_val

    return False


# ═══════════════════════════════════════════════════════════════════
#  Workflow Executor
# ═══════════════════════════════════════════════════════════════════

class WorkflowExecutor:
    """
    §10 Workflow Executor.

    Executes a DAG of workflow steps with dependency ordering,
    output passing, conditional branching, and for_each.
    """

    def __init__(
        self,
        registry: WorkflowFunctionRegistry | None = None,
    ):
        self._registry = registry or WorkflowFunctionRegistry()
        self._steps: dict[str, WorkflowStep] = {}
        self._step_order: list[str] = []
        self._log = PraxisLogger.instance()

    def add_step(self, step: WorkflowStep) -> None:
        """Add a step to the workflow."""
        if step.id in self._steps:
            raise ValueError(f"Duplicate step ID: {step.id}")
        self._steps[step.id] = step
        self._step_order.append(step.id)

    @property
    def steps(self) -> list[WorkflowStep]:
        return [self._steps[sid] for sid in self._step_order]

    @property
    def size(self) -> int:
        return len(self._steps)

    def topological_order(self) -> list[str]:
        """Compute execution order respecting dependencies."""
        in_degree: dict[str, int] = defaultdict(int)
        adjacency: dict[str, list[str]] = defaultdict(list)

        for sid, step in self._steps.items():
            if sid not in in_degree:
                in_degree[sid] = 0
            for dep in step.depends_on:
                if dep not in self._steps:
                    raise ValueError(
                        f"Step '{sid}' depends on unknown step '{dep}'"
                    )
                adjacency[dep].append(sid)
                in_degree[sid] += 1

        # Kahn's algorithm
        queue = [sid for sid in self._step_order if in_degree[sid] == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for neighbor in adjacency[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(self._steps):
            raise ValueError("Cycle detected in workflow DAG")

        return result

    def run(self) -> WorkflowResult:
        """Execute the workflow in topological order."""
        t0 = time.monotonic()
        result = WorkflowResult(steps_total=len(self._steps))
        outputs: dict[str, Any] = {}

        try:
            order = self.topological_order()
        except ValueError as e:
            result.error = str(e)
            result.total_duration = time.monotonic() - t0
            return result

        for step_id in order:
            step = self._steps[step_id]

            # Check if any dependency failed
            deps_ok = all(
                self._steps[d].status == StepStatus.COMPLETED
                for d in step.depends_on
            )
            if not deps_ok:
                step.status = StepStatus.SKIPPED
                step.error = "Dependency failed"
                result.steps_skipped += 1
                result.step_results.append(step)
                continue

            # Execute step
            step_t0 = time.monotonic()
            step.status = StepStatus.RUNNING

            try:
                step.output = self._execute_step(step, outputs)
                step.status = StepStatus.COMPLETED
                outputs[step_id] = step.output
                result.outputs[step_id] = step.output
                result.steps_completed += 1
            except Exception as e:
                step.status = StepStatus.FAILED
                step.error = str(e)
                result.steps_failed += 1
                self._log.error(
                    f"Workflow step '{step_id}' failed: {e}",
                    tags={"workflow"},
                )

            step.duration_seconds = time.monotonic() - step_t0
            result.step_results.append(step)

        result.total_duration = time.monotonic() - t0
        return result

    def _execute_step(
        self, step: WorkflowStep, outputs: dict[str, Any]
    ) -> Any:
        """Execute one step, handling conditionals and for_each."""

        # ── Conditional branching ─────────────────────────────
        if step.condition is not None:
            cond_result = _evaluate_condition(step.condition, outputs)
            branch = step.if_true if cond_result else step.if_false
            if branch is None:
                return None

            fn_name = branch.get("function")
            params = branch.get("params", {})
            return self._call_function(fn_name, params, outputs)

        # ── for_each ──────────────────────────────────────────
        if step.for_each is not None:
            items = _resolve_reference(step.for_each, outputs)
            if items is None:
                items = []

            results = []
            for item in items:
                merged_params = dict(step.params)
                merged_params[step.for_each_as] = item
                # Inject dependency outputs
                merged_params["_outputs"] = outputs
                fn_name = step.function
                result = self._call_function(fn_name, merged_params, outputs)
                results.append(result)

            return results

        # ── Normal step ───────────────────────────────────────
        return self._call_function(step.function, step.params, outputs)

    def _call_function(
        self,
        fn_name: str | None,
        params: dict[str, Any],
        outputs: dict[str, Any],
    ) -> Any:
        """Resolve and call a function."""
        if fn_name is None:
            return None

        fn = self._registry.resolve(fn_name)
        if fn is None:
            raise ValueError(f"Unknown function: {fn_name}")

        # Resolve any parameter references (strings starting with step IDs)
        resolved_params = {}
        for k, v in params.items():
            if k == "_outputs":
                resolved_params[k] = v
            elif isinstance(v, str) and "." in v and v.split(".")[0] in outputs:
                resolved_params[k] = _resolve_reference(v, outputs)
            else:
                resolved_params[k] = v

        return fn(**resolved_params)


# ═══════════════════════════════════════════════════════════════════
#  YAML Loader
# ═══════════════════════════════════════════════════════════════════

def workflow_from_config(
    config: dict[str, Any],
    registry: WorkflowFunctionRegistry,
) -> WorkflowExecutor:
    """
    Build a WorkflowExecutor from a YAML workflow config block.

    Config format matches §10:
        workflow:
          steps:
            - id: step_a
              function: my_func
              params: {x: 1}
            - id: step_b
              depends_on: [step_a]
              condition: "step_a.output.count > 100"
              if_true: {function: do_thing}
              if_false: {function: do_other}
    """
    executor = WorkflowExecutor(registry=registry)
    workflow_block = config.get("workflow", config)
    steps = workflow_block.get("steps", [])

    for step_def in steps:
        step = WorkflowStep(
            id=step_def["id"],
            function=step_def.get("function"),
            params=step_def.get("params", {}),
            depends_on=step_def.get("depends_on", []),
            condition=step_def.get("condition"),
            if_true=step_def.get("if_true"),
            if_false=step_def.get("if_false"),
            for_each=step_def.get("for_each"),
            for_each_as=step_def.get("as", "item"),
            parallel=step_def.get("parallel", False),
        )
        executor.add_step(step)

    return executor
