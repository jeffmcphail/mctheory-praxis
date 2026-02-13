"""
Port Validation (Phase 3.8).

Compare every intermediate output of the Praxis CPO pipeline
against the original pair_trade_gld_gdx() logic.

Validation stages (per the 5% debugging strategy):
1. Z-score signal computation (reference: pandas ewm)
2. Position signal entry/exit (reference: pandas vectorized assign)
3. P&L calculation
4. Sharpe ratio and metrics

Each stage returns a ValidationResult with match status and diffs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from praxis.cpo import execute_single_leg
from praxis.signals.zscore import _ewm_mean, _ewm_std


@dataclass
class StageResult:
    """Result of one validation stage."""
    stage: str
    passed: bool
    max_diff: float
    mean_diff: float
    detail: str = ""


@dataclass
class PortValidationResult:
    """Full port validation across all stages."""
    stages: list[StageResult] = field(default_factory=list)
    all_passed: bool = False
    summary: str = ""

    def add(self, stage: StageResult):
        self.stages.append(stage)
        self.all_passed = all(s.passed for s in self.stages)


def validate_zscore_port(
    close_a: np.ndarray,
    close_b: np.ndarray,
    weight: float,
    lookback: int,
    tolerance: float = 1e-8,
) -> StageResult:
    """
    Stage 1: Validate z-score computation matches original pandas.

    Reference: pandas ewm(span=lookback, adjust=False) for mean and std.
    """
    spread = close_a - weight * close_b

    # Praxis computation
    ema_mean = _ewm_mean(spread, lookback)
    ema_std = _ewm_std(spread, lookback)
    with np.errstate(invalid="ignore"):
        praxis_zscore = np.where(ema_std > 0, (spread - ema_mean) / ema_std, 0.0)

    # Reference: pandas
    pd_spread = pd.Series(spread)
    pd_mean = pd_spread.ewm(span=lookback, adjust=False).mean().values
    pd_std = pd_spread.ewm(span=lookback, adjust=False).std().values
    with np.errstate(invalid="ignore"):
        ref_zscore = np.where(
            (~np.isnan(pd_std)) & (pd_std > 0),
            (spread - pd_mean) / pd_std,
            0.0,
        )

    # Compare where both are valid
    valid = (ema_std > 0) & (~np.isnan(pd_std)) & (pd_std > 0)
    if valid.sum() > 0:
        diff = np.abs(praxis_zscore[valid] - ref_zscore[valid])
        max_diff = float(np.max(diff))
        mean_diff = float(np.mean(diff))
    else:
        max_diff = 0.0
        mean_diff = 0.0

    passed = max_diff < tolerance
    return StageResult(
        stage="zscore_computation",
        passed=passed,
        max_diff=max_diff,
        mean_diff=mean_diff,
        detail=f"Compared {valid.sum()} valid bars",
    )


def validate_positions_port(
    close_a: np.ndarray,
    close_b: np.ndarray,
    weight: float,
    lookback: int,
    entry_threshold: float,
    exit_threshold_fraction: float,
    tolerance: float = 1e-10,
) -> StageResult:
    """
    Stage 2: Validate position entry/exit matches original.

    Original code: vectorized threshold assignment, then ffill.
    Since positions are initialized to 0 (not NaN), ffill is a no-op.
    Each bar is independently determined by z-score thresholds.
    """
    n = len(close_a)
    spread = close_a - weight * close_b
    exit_threshold = exit_threshold_fraction * entry_threshold

    # Use pandas as reference (exact original logic)
    pd_spread = pd.Series(spread)
    pd_mean = pd_spread.ewm(span=lookback, adjust=False).mean()
    pd_std = pd_spread.ewm(span=lookback, adjust=False).std()
    pd_zscore = (pd_spread - pd_mean) / pd_std

    ref_long = pd.Series(np.zeros(n))
    ref_short = pd.Series(np.zeros(n))

    ref_short[pd_zscore >= entry_threshold] = -1
    ref_long[pd_zscore <= -entry_threshold] = 1
    ref_short[pd_zscore <= exit_threshold] = 0
    ref_long[pd_zscore >= -exit_threshold] = 0

    ref_combined = (ref_long + ref_short).values

    # Praxis
    open_a = close_a.copy()
    slr = execute_single_leg(
        close_a, open_a, close_b,
        {"weight": weight, "lookback": lookback,
         "entry_threshold": entry_threshold,
         "exit_threshold_fraction": exit_threshold_fraction},
        transaction_costs=0.0,
    )

    diff = np.abs(slr.positions - ref_combined)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    mismatches = int(np.sum(diff > tolerance))

    return StageResult(
        stage="position_signals",
        passed=mismatches == 0,
        max_diff=max_diff,
        mean_diff=mean_diff,
        detail=f"{mismatches} mismatches out of {n} bars",
    )


def validate_pnl_port(
    close_a: np.ndarray,
    open_a: np.ndarray,
    close_b: np.ndarray,
    params: dict[str, Any],
    transaction_costs: float = 0.0005,
    tolerance: float = 1e-6,
) -> StageResult:
    """
    Stage 3: Validate P&L calculation matches original.

    Original: pnl = positions.shift() * (close_GLD - open_GLD) / open_GLD
    """
    slr = execute_single_leg(close_a, open_a, close_b, params,
                              transaction_costs=transaction_costs)

    positions = slr.positions
    n = len(positions)
    with np.errstate(invalid="ignore", divide="ignore"):
        period_return = np.where(open_a != 0, (close_a - open_a) / open_a, 0.0)

    shifted_pos = np.zeros(n)
    shifted_pos[1:] = positions[:-1]
    ref_pnl = shifted_pos * period_return
    pos_diff = np.diff(positions, prepend=0)
    ref_tc = np.abs(pos_diff) * transaction_costs
    ref_pnl_tc = ref_pnl - ref_tc

    diff = np.abs(slr.pnl - ref_pnl_tc)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))

    return StageResult(
        stage="pnl_calculation",
        passed=max_diff < tolerance,
        max_diff=max_diff,
        mean_diff=mean_diff,
        detail=f"daily_return={slr.daily_return:.8f}",
    )


def validate_metrics_port(
    close_a: np.ndarray,
    open_a: np.ndarray,
    close_b: np.ndarray,
    params: dict[str, Any],
    tolerance: float = 1e-6,
) -> StageResult:
    """
    Stage 4: Validate Sharpe and other metrics match original formulas.
    """
    periods_per_year = 252 * 6.5 * 60
    slr = execute_single_leg(close_a, open_a, close_b, params,
                              transaction_costs=0.0005,
                              periods_per_year=periods_per_year)

    pnl_slice = slr.pnl[1:]
    ref_mean = float(np.mean(pnl_slice)) if len(pnl_slice) > 0 else 0.0
    ref_std = float(np.std(pnl_slice)) if len(pnl_slice) > 0 else 0.0

    if ref_std > 0:
        ref_sharpe = float(np.sqrt(periods_per_year) * ref_mean / ref_std)
    else:
        ref_sharpe = 0.0

    sharpe_diff = abs(slr.sharpe_ratio - ref_sharpe)

    return StageResult(
        stage="metrics_computation",
        passed=sharpe_diff < tolerance,
        max_diff=sharpe_diff,
        mean_diff=sharpe_diff,
        detail=f"praxis_sharpe={slr.sharpe_ratio:.6f}, ref_sharpe={ref_sharpe:.6f}",
    )


def run_full_validation(
    close_a: np.ndarray,
    open_a: np.ndarray,
    close_b: np.ndarray,
    params: dict[str, Any],
) -> PortValidationResult:
    """Run all 4 validation stages."""
    result = PortValidationResult()

    result.add(validate_zscore_port(
        close_a, close_b, params["weight"], params["lookback"],
    ))
    result.add(validate_positions_port(
        close_a, close_b, params["weight"], params["lookback"],
        params["entry_threshold"],
        params.get("exit_threshold_fraction", -0.6),
    ))
    result.add(validate_pnl_port(close_a, open_a, close_b, params))
    result.add(validate_metrics_port(close_a, open_a, close_b, params))

    result.summary = (
        f"{'PASS' if result.all_passed else 'FAIL'}: "
        f"{sum(1 for s in result.stages if s.passed)}/{len(result.stages)} stages passed"
    )
    return result
