"""
Backtest Reconciliation (Phase 3.10, §9.2).

Compare vectorized and event-driven backtest results on the same
positions/prices to ensure they agree within tolerance.

Key insight: vectorized assumes trade-on-next-bar at close price.
Event-driven with fill_on="close" should match closely. With
fill_on="next_open" there will be fill-timing differences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from praxis.backtest import BacktestOutput, VectorizedEngine
from praxis.backtest.event_driven import EventDrivenEngine
from praxis.logger.core import PraxisLogger


@dataclass
class ReconciliationResult:
    """Result of comparing two backtest engines."""
    vectorized: BacktestOutput
    event_driven: BacktestOutput
    total_return_diff: float
    sharpe_diff: float
    max_drawdown_diff: float
    equity_rmse: float
    equity_max_diff: float
    position_mismatches: int
    within_tolerance: bool
    tolerance: float
    details: dict[str, Any]


def reconcile(
    positions: pl.Series | np.ndarray,
    prices: pl.Series | np.ndarray,
    tolerance: float = 0.05,
    initial_capital: float = 1.0,
) -> ReconciliationResult:
    """
    Run both engines on same inputs and compare.

    The event-driven engine is configured with fill_on="close" to
    best match vectorized assumptions. Even so, small differences
    arise from the explicit cash/position tracking.

    Args:
        positions: Signal/position series.
        prices: Close price series.
        tolerance: Max allowed difference for key metrics (fraction).
        initial_capital: Starting capital.

    Returns:
        ReconciliationResult with comparison details.
    """
    log = PraxisLogger.instance()

    # ── Run vectorized ────────────────────────────────────────
    vec_engine = VectorizedEngine()
    vec_out = vec_engine.run(positions, prices, initial_capital=initial_capital)

    # ── Run event-driven (fill at close to match vectorized) ──
    ed_engine = EventDrivenEngine(fill_on="close")
    ed_out = ed_engine.run(positions, prices, initial_capital=initial_capital)

    # ── Compare metrics ───────────────────────────────────────
    v = vec_out.metrics
    e = ed_out.metrics

    ret_diff = abs(v.total_return - e.total_return)
    sharpe_diff = abs(v.sharpe_ratio - e.sharpe_ratio)
    dd_diff = abs(v.max_drawdown - e.max_drawdown)

    # ── Compare equity curves ─────────────────────────────────
    # They may differ in length by 1 due to convention differences
    min_len = min(len(vec_out.equity_curve), len(ed_out.equity_curve))
    vec_eq = vec_out.equity_curve[:min_len]
    ed_eq = ed_out.equity_curve[:min_len]

    # Normalize both to start at 1.0 for fair comparison
    if vec_eq[0] != 0:
        vec_norm = vec_eq / vec_eq[0]
    else:
        vec_norm = vec_eq
    if ed_eq[0] != 0:
        ed_norm = ed_eq / ed_eq[0]
    else:
        ed_norm = ed_eq

    eq_diff = vec_norm - ed_norm
    equity_rmse = float(np.sqrt(np.mean(eq_diff ** 2)))
    equity_max_diff = float(np.max(np.abs(eq_diff)))

    # ── Compare positions ─────────────────────────────────────
    min_pos_len = min(len(vec_out.positions), len(ed_out.positions))
    pos_mismatches = int(np.sum(
        vec_out.positions[:min_pos_len] != ed_out.positions[:min_pos_len]
    ))

    # ── Determine pass/fail ───────────────────────────────────
    within = (ret_diff <= tolerance and sharpe_diff <= tolerance * 10)

    details = {
        "vec_total_return": v.total_return,
        "ed_total_return": e.total_return,
        "vec_sharpe": v.sharpe_ratio,
        "ed_sharpe": e.sharpe_ratio,
        "vec_max_dd": v.max_drawdown,
        "ed_max_dd": e.max_drawdown,
        "vec_trades": v.total_trades,
        "ed_trades": e.total_trades,
        "equity_curve_len_vec": len(vec_out.equity_curve),
        "equity_curve_len_ed": len(ed_out.equity_curve),
    }

    log.info(
        f"Reconciliation: return_diff={ret_diff:.6f}, "
        f"sharpe_diff={sharpe_diff:.4f}, dd_diff={dd_diff:.6f}, "
        f"eq_rmse={equity_rmse:.6f}, pos_mismatches={pos_mismatches}, "
        f"within_tolerance={within}",
        tags={"backtest", "reconciliation"},
    )

    return ReconciliationResult(
        vectorized=vec_out,
        event_driven=ed_out,
        total_return_diff=ret_diff,
        sharpe_diff=sharpe_diff,
        max_drawdown_diff=dd_diff,
        equity_rmse=equity_rmse,
        equity_max_diff=equity_max_diff,
        position_mismatches=pos_mismatches,
        within_tolerance=within,
        tolerance=tolerance,
        details=details,
    )
