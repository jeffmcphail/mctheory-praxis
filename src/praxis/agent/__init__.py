"""
Agent Autonomy Framework (Phase 4.10, §11).

Orchestrates the overnight autonomous flow:
1. Discovery — scan sources for new strategies/ideas
2. Construction — parse methodologies into model configs
3. Backtesting — run backtests on promising candidates
4. Reporting — generate summary ranked by performance
5. Delivery — email/dashboard/log

The framework is event-driven and checkpoint-able: if a step fails,
the pipeline can resume from the last successful checkpoint.

Usage:
    agent = AutonomousAgent(config)
    result = agent.run_overnight_cycle(price_data)
    print(result.report)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

import numpy as np

from praxis.logger.core import PraxisLogger


# ═══════════════════════════════════════════════════════════════════
#  Types
# ═══════════════════════════════════════════════════════════════════

class AgentPhase(str, Enum):
    IDLE = "idle"
    DISCOVERY = "discovery"
    CONSTRUCTION = "construction"
    BACKTESTING = "backtesting"
    REPORTING = "reporting"
    DELIVERY = "delivery"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class AgentConfig:
    """Configuration for the autonomous agent (§11.1)."""
    # Discovery
    discovery_enabled: bool = True
    discovery_sources: list[dict[str, Any]] = field(default_factory=lambda: [
        {"type": "scan", "method": "burgess"},
    ])

    # Backtesting
    backtest_enabled: bool = True
    max_concurrent_backtests: int = 4
    max_compute_seconds: float = 300.0

    # Reporting
    report_enabled: bool = True
    min_sharpe_for_report: float = 0.5
    top_n_report: int = 10

    # Delivery
    delivery_methods: list[str] = field(default_factory=lambda: ["log"])

    # Approval gates
    approval_required_for: list[str] = field(default_factory=lambda: ["live_trading"])

    # Notifications
    notify_on: list[str] = field(default_factory=lambda: [
        "backtest_exceptional", "error_critical",
    ])
    sharpe_alert_threshold: float = 2.0
    drawdown_alert_threshold: float = 0.15


@dataclass
class DiscoveryResult:
    """Result of the discovery phase."""
    source: str = ""
    method: str = ""
    candidates: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


@dataclass
class BacktestCandidate:
    """A candidate from discovery ready for backtesting."""
    candidate_id: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    description: str = ""


@dataclass
class BacktestOutcome:
    """Result of backtesting a single candidate."""
    candidate_id: str = ""
    sharpe_ratio: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    n_trades: int = 0
    win_rate: float = 0.0
    elapsed_seconds: float = 0.0
    success: bool = True
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentReport:
    """Summary report from an overnight cycle."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    n_discovered: int = 0
    n_backtested: int = 0
    n_exceptional: int = 0
    top_results: list[BacktestOutcome] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)
    total_elapsed_seconds: float = 0.0
    phases_completed: list[str] = field(default_factory=list)

    def format_text(self) -> str:
        """Format report as human-readable text."""
        lines = [
            f"═══ Autonomous Agent Report ═══",
            f"Time: {self.timestamp.isoformat()}",
            f"Discovered: {self.n_discovered} | Backtested: {self.n_backtested} | Exceptional: {self.n_exceptional}",
            f"Elapsed: {self.total_elapsed_seconds:.1f}s",
            "",
        ]

        if self.alerts:
            lines.append("⚠ Alerts:")
            for alert in self.alerts:
                lines.append(f"  • {alert}")
            lines.append("")

        if self.top_results:
            lines.append("Top Results:")
            for i, r in enumerate(self.top_results, 1):
                lines.append(
                    f"  {i}. {r.candidate_id}: "
                    f"Sharpe={r.sharpe_ratio:.2f}, "
                    f"Return={r.total_return:.1%}, "
                    f"MaxDD={r.max_drawdown:.1%}, "
                    f"Trades={r.n_trades}"
                )
            lines.append("")

        lines.append(f"Phases: {' → '.join(self.phases_completed)}")
        return "\n".join(lines)


@dataclass
class CycleResult:
    """Full result of an overnight cycle."""
    success: bool = True
    phase: AgentPhase = AgentPhase.COMPLETE
    discoveries: list[DiscoveryResult] = field(default_factory=list)
    backtests: list[BacktestOutcome] = field(default_factory=list)
    report: AgentReport = field(default_factory=AgentReport)
    error: str = ""


# ═══════════════════════════════════════════════════════════════════
#  Agent
# ═══════════════════════════════════════════════════════════════════

class AutonomousAgent:
    """
    Autonomous overnight agent (§11).

    Pluggable discovery and backtest functions allow different
    strategies to be scanned without changing the framework.
    """

    def __init__(self, config: AgentConfig | None = None):
        self._config = config or AgentConfig()
        self._log = PraxisLogger.instance()
        self._phase = AgentPhase.IDLE
        self._discovery_fns: list[Callable] = []
        self._backtest_fn: Callable | None = None
        self._delivery_fns: dict[str, Callable] = {}

    @property
    def config(self) -> AgentConfig:
        return self._config

    @property
    def phase(self) -> AgentPhase:
        return self._phase

    def register_discovery(self, fn: Callable[..., DiscoveryResult]) -> None:
        """Register a discovery function."""
        self._discovery_fns.append(fn)

    def register_backtest(self, fn: Callable[[BacktestCandidate], BacktestOutcome]) -> None:
        """Register the backtest execution function."""
        self._backtest_fn = fn

    def register_delivery(self, method: str, fn: Callable[[AgentReport], None]) -> None:
        """Register a delivery method (email, dashboard, log, etc.)."""
        self._delivery_fns[method] = fn

    def run_overnight_cycle(
        self,
        price_data: np.ndarray | None = None,
        **kwargs,
    ) -> CycleResult:
        """
        Execute the full overnight cycle.

        Args:
            price_data: Optional price matrix for backtest.
            **kwargs: Passed to discovery and backtest functions.

        Returns:
            CycleResult with all phase outputs.
        """
        t0 = time.monotonic()
        result = CycleResult()
        cfg = self._config

        try:
            # Phase 1: Discovery
            if cfg.discovery_enabled:
                self._phase = AgentPhase.DISCOVERY
                self._log.info("Agent: starting discovery phase", tags={"agent"})
                discoveries = self._run_discovery(**kwargs)
                result.discoveries = discoveries
                result.report.n_discovered = sum(len(d.candidates) for d in discoveries)
                result.report.phases_completed.append("discovery")

            # Phase 2: Construction (extract backtest candidates)
            self._phase = AgentPhase.CONSTRUCTION
            candidates = self._construct_candidates(result.discoveries)
            result.report.phases_completed.append("construction")

            # Phase 3: Backtesting
            if cfg.backtest_enabled and self._backtest_fn is not None:
                self._phase = AgentPhase.BACKTESTING
                self._log.info(
                    f"Agent: backtesting {len(candidates)} candidates",
                    tags={"agent"},
                )
                outcomes = self._run_backtests(candidates, price_data=price_data, **kwargs)
                result.backtests = outcomes
                result.report.n_backtested = len(outcomes)
                result.report.phases_completed.append("backtesting")

            # Phase 4: Reporting
            if cfg.report_enabled:
                self._phase = AgentPhase.REPORTING
                self._build_report(result)
                result.report.phases_completed.append("reporting")

            # Phase 5: Delivery
            self._phase = AgentPhase.DELIVERY
            self._deliver_report(result.report)
            result.report.phases_completed.append("delivery")

            self._phase = AgentPhase.COMPLETE
            result.success = True

        except Exception as e:
            self._phase = AgentPhase.FAILED
            result.success = False
            result.error = str(e)
            result.phase = AgentPhase.FAILED
            self._log.error(f"Agent: cycle failed — {e}", tags={"agent"})

        result.report.total_elapsed_seconds = time.monotonic() - t0
        return result

    # ── Phase Implementations ─────────────────────────────────

    def _run_discovery(self, **kwargs) -> list[DiscoveryResult]:
        """Execute all registered discovery functions."""
        results = []
        for fn in self._discovery_fns:
            t0 = time.monotonic()
            try:
                r = fn(**kwargs)
                r.elapsed_seconds = time.monotonic() - t0
                results.append(r)
            except Exception as e:
                self._log.warning(f"Agent: discovery function failed — {e}", tags={"agent"})
                results.append(DiscoveryResult(
                    source="error",
                    metadata={"error": str(e)},
                    elapsed_seconds=time.monotonic() - t0,
                ))
        return results

    def _construct_candidates(
        self, discoveries: list[DiscoveryResult],
    ) -> list[BacktestCandidate]:
        """Extract backtest candidates from discovery results."""
        candidates = []
        for disc in discoveries:
            for i, cand in enumerate(disc.candidates):
                candidates.append(BacktestCandidate(
                    candidate_id=cand.get("id", f"{disc.source}_{i}"),
                    config=cand.get("config", {}),
                    source=disc.source,
                    description=cand.get("description", ""),
                ))
        return candidates

    def _run_backtests(
        self,
        candidates: list[BacktestCandidate],
        **kwargs,
    ) -> list[BacktestOutcome]:
        """Run backtests on candidates with resource limits."""
        outcomes = []
        deadline = time.monotonic() + self._config.max_compute_seconds

        for cand in candidates:
            if time.monotonic() > deadline:
                self._log.warning("Agent: compute budget exhausted", tags={"agent"})
                break

            t0 = time.monotonic()
            try:
                outcome = self._backtest_fn(cand, **kwargs)
                outcome.elapsed_seconds = time.monotonic() - t0
                outcomes.append(outcome)
            except Exception as e:
                outcomes.append(BacktestOutcome(
                    candidate_id=cand.candidate_id,
                    success=False,
                    error=str(e),
                    elapsed_seconds=time.monotonic() - t0,
                ))

        return outcomes

    def _build_report(self, result: CycleResult) -> None:
        """Build the summary report with ranking and alerts."""
        cfg = self._config

        # Filter successful backtests
        successful = [b for b in result.backtests if b.success]

        # Rank by Sharpe
        successful.sort(key=lambda b: b.sharpe_ratio, reverse=True)

        # Top N
        result.report.top_results = successful[:cfg.top_n_report]

        # Exceptional count
        exceptional = [b for b in successful if b.sharpe_ratio >= cfg.min_sharpe_for_report]
        result.report.n_exceptional = len(exceptional)

        # Alerts
        for b in successful:
            if b.sharpe_ratio >= cfg.sharpe_alert_threshold:
                result.report.alerts.append(
                    f"High Sharpe: {b.candidate_id} = {b.sharpe_ratio:.2f}"
                )
            if b.max_drawdown >= cfg.drawdown_alert_threshold:
                result.report.alerts.append(
                    f"High Drawdown: {b.candidate_id} = {b.max_drawdown:.1%}"
                )

    def _deliver_report(self, report: AgentReport) -> None:
        """Deliver report through configured methods."""
        for method in self._config.delivery_methods:
            fn = self._delivery_fns.get(method)
            if fn is not None:
                try:
                    fn(report)
                except Exception as e:
                    self._log.warning(
                        f"Agent: delivery via {method} failed — {e}",
                        tags={"agent"},
                    )
            elif method == "log":
                # Default: log the report
                self._log.info(
                    f"Agent report:\n{report.format_text()}",
                    tags={"agent"},
                )
