"""
Crypto Cointegration + Flash Loan Orchestrator (Phase 5.6 + 5.9).

Adapts the Burgess statistical arbitrage framework for crypto:
- Shorter lookback windows (crypto mean-reversion is faster)
- Higher frequency data (10s - 1min intervals)
- Z-score signal generation tuned for DeFi execution
- Flash loan orchestrator: signal → simulate → execute pipeline

Usage:
    analyzer = CryptoCointegrationAnalyzer(config)
    signal = analyzer.compute_signal(prices_a, prices_b)
    if signal.should_trade:
        orchestrator.execute_opportunity(signal)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from praxis.logger.core import PraxisLogger


# ═══════════════════════════════════════════════════════════════════
#  Crypto Cointegration Config
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CryptoCointegrationConfig:
    """
    Configuration tuned for crypto markets (§14.5 construction).

    Key differences from equity stat arb:
    - Shorter windows (crypto regimes change faster)
    - Higher frequency (10s to 1min)
    - Wider z-score thresholds (more volatile)
    """
    # Lookback
    zscore_window: int = 30          # vs 60+ for equities
    half_life_window: int = 60
    min_history_points: int = 12     # Minimum for ADF validity

    # Entry/exit
    z_score_entry: float = 2.0
    z_score_exit: float = 0.5
    z_score_emergency: float = 4.0   # Force close

    # Cointegration filters
    adf_pvalue: float = 0.05
    min_correlation: float = 0.7
    max_half_life: float = 50.0      # In periods (bars)
    min_half_life: float = 2.0

    # Signal
    signal_decay_periods: int = 10   # Signal weakens over time
    cooldown_periods: int = 3        # Min periods between signals


@dataclass
class CryptoSignal:
    """A trading signal from the crypto cointegration analyzer."""
    pair: str = ""
    asset_a: str = ""
    asset_b: str = ""
    z_score: float = 0.0
    half_life: float = 0.0
    hedge_ratio: float = 0.0
    spread: float = 0.0
    signal_strength: float = 0.0     # 0 to 1
    direction: str = ""              # "long_a_short_b", "short_a_long_b", "flat"
    should_trade: bool = False
    should_close: bool = False
    is_emergency: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_entry(self) -> bool:
        return self.should_trade and not self.should_close

    @property
    def is_exit(self) -> bool:
        return self.should_close


@dataclass
class CointPairResult:
    """Result of cointegration analysis for a pair."""
    asset_a: str = ""
    asset_b: str = ""
    is_cointegrated: bool = False
    adf_pvalue: float = 1.0
    correlation: float = 0.0
    hedge_ratio: float = 0.0
    half_life: float = 0.0
    mean_spread: float = 0.0
    std_spread: float = 0.0
    n_observations: int = 0


# ═══════════════════════════════════════════════════════════════════
#  Crypto Cointegration Analyzer
# ═══════════════════════════════════════════════════════════════════

class CryptoCointegrationAnalyzer:
    """
    Cointegration analyzer adapted for crypto (§14.5).

    Uses OLS for hedge ratio, Dickey-Fuller for stationarity,
    and half-life estimation for mean-reversion speed.
    """

    def __init__(self, config: CryptoCointegrationConfig | None = None):
        self._config = config or CryptoCointegrationConfig()
        self._log = PraxisLogger.instance()
        self._last_signal_idx: dict[str, int] = {}
        self._bar_count = 0

    @property
    def config(self) -> CryptoCointegrationConfig:
        return self._config

    def test_cointegration(
        self,
        prices_a: np.ndarray,
        prices_b: np.ndarray,
        asset_a: str = "A",
        asset_b: str = "B",
    ) -> CointPairResult:
        """
        Test whether two price series are cointegrated.

        Uses OLS regression for hedge ratio, then ADF on residuals.
        """
        prices_a = np.asarray(prices_a, dtype=float).ravel()
        prices_b = np.asarray(prices_b, dtype=float).ravel()
        n = min(len(prices_a), len(prices_b))
        result = CointPairResult(asset_a=asset_a, asset_b=asset_b, n_observations=n)

        if n < self._config.min_history_points:
            return result

        a = prices_a[:n]
        b = prices_b[:n]

        # Correlation
        result.correlation = float(np.corrcoef(a, b)[0, 1])
        if abs(result.correlation) < self._config.min_correlation:
            return result

        # OLS hedge ratio: a = beta * b + residual
        b_mean = np.mean(b)
        a_mean = np.mean(a)
        cov = np.sum((b - b_mean) * (a - a_mean))
        var_b = np.sum((b - b_mean) ** 2)
        if var_b < 1e-10:
            return result
        beta = cov / var_b
        result.hedge_ratio = float(beta)

        # Spread (residuals)
        spread = a - beta * b
        result.mean_spread = float(np.mean(spread))
        result.std_spread = float(np.std(spread))

        # ADF test on spread (simplified: check if spread is mean-reverting)
        if len(spread) >= 10:
            result.adf_pvalue = self._simple_adf(spread)

        result.is_cointegrated = result.adf_pvalue < self._config.adf_pvalue

        # Half-life
        if result.is_cointegrated and result.std_spread > 1e-10:
            result.half_life = self._estimate_half_life(spread)

        return result

    def compute_signal(
        self,
        prices_a: np.ndarray,
        prices_b: np.ndarray,
        hedge_ratio: float | None = None,
        asset_a: str = "A",
        asset_b: str = "B",
    ) -> CryptoSignal:
        """
        Compute current trading signal from z-score.

        If hedge_ratio not provided, estimates from data.
        """
        prices_a = np.asarray(prices_a, dtype=float).ravel()
        prices_b = np.asarray(prices_b, dtype=float).ravel()
        n = min(len(prices_a), len(prices_b))
        pair = f"{asset_a}/{asset_b}"

        signal = CryptoSignal(pair=pair, asset_a=asset_a, asset_b=asset_b)

        if n < self._config.zscore_window + 1:
            return signal

        a = prices_a[:n]
        b = prices_b[:n]

        # Estimate hedge ratio if not provided
        if hedge_ratio is None:
            window = a[-self._config.zscore_window:]
            window_b = b[-self._config.zscore_window:]
            var_b = np.var(window_b)
            if var_b < 1e-10:
                return signal
            hedge_ratio = float(np.cov(window, window_b)[0, 1] / var_b)

        signal.hedge_ratio = hedge_ratio

        # Compute spread and z-score over window
        spread = a - hedge_ratio * b
        window_spread = spread[-self._config.zscore_window:]
        mu = np.mean(window_spread)
        sigma = np.std(window_spread)

        if sigma < 1e-10:
            return signal

        current_spread = spread[-1]
        z = (current_spread - mu) / sigma
        signal.z_score = float(z)
        signal.spread = float(current_spread)

        # Half-life
        if len(spread) >= 10:
            signal.half_life = self._estimate_half_life(spread[-self._config.half_life_window:])

        # Direction and entry/exit logic
        self._bar_count += 1
        cfg = self._config

        # Emergency close
        if abs(z) >= cfg.z_score_emergency:
            signal.is_emergency = True
            signal.should_close = True
            signal.direction = "flat"
            signal.signal_strength = 1.0
            return signal

        # Exit signal
        if abs(z) <= cfg.z_score_exit:
            signal.should_close = True
            signal.direction = "flat"
            signal.signal_strength = 0.0
            return signal

        # Entry signal
        if abs(z) >= cfg.z_score_entry:
            # Check cooldown
            last = self._last_signal_idx.get(pair, -cfg.cooldown_periods - 1)
            if self._bar_count - last >= cfg.cooldown_periods:
                signal.should_trade = True
                signal.signal_strength = min(1.0, (abs(z) - cfg.z_score_entry) / 2.0)
                self._last_signal_idx[pair] = self._bar_count

                if z >= cfg.z_score_entry:
                    signal.direction = "short_a_long_b"
                else:
                    signal.direction = "long_a_short_b"

        return signal

    def scan_pairs(
        self,
        price_matrix: dict[str, np.ndarray],
    ) -> list[CointPairResult]:
        """
        Scan all pairs for cointegration.

        Args:
            price_matrix: {asset_name: price_array}

        Returns:
            Cointegrated pairs sorted by p-value.
        """
        assets = list(price_matrix.keys())
        results = []

        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                result = self.test_cointegration(
                    price_matrix[assets[i]],
                    price_matrix[assets[j]],
                    asset_a=assets[i],
                    asset_b=assets[j],
                )
                if result.is_cointegrated:
                    results.append(result)

        results.sort(key=lambda r: r.adf_pvalue)
        return results

    def _simple_adf(self, series: np.ndarray) -> float:
        """
        Simplified ADF test.

        Regresses diff(y) on lag(y) and checks t-statistic.
        Returns approximate p-value (uses critical value thresholds).
        """
        y = series.ravel()
        n = len(y)
        if n < 10:
            return 1.0

        dy = np.diff(y)
        y_lag = y[:-1]

        # Regress dy = alpha + beta * y_lag + epsilon
        x = np.column_stack([np.ones(len(y_lag)), y_lag])
        try:
            beta = np.linalg.lstsq(x, dy, rcond=None)[0]
            residuals = dy - x @ beta
            se = np.sqrt(np.sum(residuals ** 2) / (len(dy) - 2))
            x_inv = np.linalg.inv(x.T @ x)
            t_stat = beta[1] / (se * np.sqrt(x_inv[1, 1]))
        except (np.linalg.LinAlgError, ZeroDivisionError):
            return 1.0

        # Approximate p-value from ADF critical values (n=100)
        # 1%: -3.51, 5%: -2.89, 10%: -2.58
        t = float(t_stat)
        if t < -3.51:
            return 0.01
        elif t < -2.89:
            return 0.05
        elif t < -2.58:
            return 0.10
        else:
            return 0.50

    def _estimate_half_life(self, spread: np.ndarray) -> float:
        """Estimate mean-reversion half-life via OLS on lagged spread."""
        spread = spread.ravel()
        if len(spread) < 5:
            return float('inf')

        y = np.diff(spread)
        x = spread[:-1].reshape(-1, 1)

        try:
            beta = np.linalg.lstsq(
                np.column_stack([np.ones(len(x)), x]), y, rcond=None
            )[0]
            lam = beta[1]
            if lam >= 0:
                return float('inf')
            half_life = -np.log(2) / lam
            return max(0.0, float(half_life))
        except (np.linalg.LinAlgError, ZeroDivisionError):
            return float('inf')


# ═══════════════════════════════════════════════════════════════════
#  Flash Loan Orchestrator (Phase 5.9)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class OrchestratorConfig:
    """Configuration for the flash loan orchestrator."""
    min_profit_usd: float = 50.0
    max_gas_spend_per_day_usd: float = 100.0
    max_failed_attempts_per_hour: int = 500
    cooldown_after_success_seconds: float = 30.0
    cooldown_after_failure_seconds: float = 5.0
    borrow_amount_usd: float = 1_000_000.0
    provider: str = "balancer"


@dataclass
class OrchestratorState:
    """Runtime state for the orchestrator."""
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    total_profit_usd: float = 0.0
    total_gas_spent_usd: float = 0.0
    gas_spent_today_usd: float = 0.0
    failed_this_hour: int = 0
    last_success_time: datetime | None = None
    last_attempt_time: datetime | None = None
    is_paused: bool = False
    pause_reason: str = ""

    @property
    def net_profit(self) -> float:
        return self.total_profit_usd - self.total_gas_spent_usd

    @property
    def success_rate(self) -> float:
        if self.total_attempts == 0:
            return 0.0
        return self.successful_attempts / self.total_attempts

    @property
    def can_trade(self) -> bool:
        return not self.is_paused


class FlashLoanOrchestrator:
    """
    Full-stack flash loan arbitrage orchestrator (§14.6).

    Pipeline: signal detection → opportunity scanning → simulation
    → execution → tracking.

    Integrates:
    - CryptoCointegrationAnalyzer for signal generation
    - Opportunity scanner for venue comparison
    - Gas oracle for profitability checks
    - MEV protection for transaction submission
    """

    def __init__(self, config: OrchestratorConfig | None = None):
        self._config = config or OrchestratorConfig()
        self._state = OrchestratorState()
        self._log = PraxisLogger.instance()
        self._execute_fn = None  # Pluggable execution

    @property
    def config(self) -> OrchestratorConfig:
        return self._config

    @property
    def state(self) -> OrchestratorState:
        return self._state

    def set_execute_fn(self, fn) -> None:
        """Set custom execution function for testing."""
        self._execute_fn = fn

    def evaluate_opportunity(
        self,
        signal: CryptoSignal,
        gas_cost_usd: float = 0.10,
        flash_fee_usd: float = 0.0,
    ) -> dict[str, Any]:
        """
        Evaluate whether a signal should be acted on.

        Returns evaluation dict with go/no-go decision.
        """
        # Check if we can trade
        if not self._state.can_trade:
            return {"go": False, "reason": f"Paused: {self._state.pause_reason}"}

        # Gas budget check
        if self._state.gas_spent_today_usd >= self._config.max_gas_spend_per_day_usd:
            self._state.is_paused = True
            self._state.pause_reason = "Daily gas budget exhausted"
            return {"go": False, "reason": "Gas budget exhausted"}

        # Failure rate check
        if self._state.failed_this_hour >= self._config.max_failed_attempts_per_hour:
            return {"go": False, "reason": "Hourly failure limit reached"}

        # Signal strength check
        if not signal.should_trade:
            return {"go": False, "reason": "No entry signal"}

        # Profitability estimate
        estimated_profit = signal.signal_strength * self._config.borrow_amount_usd * 0.001
        total_cost = gas_cost_usd + flash_fee_usd
        net = estimated_profit - total_cost

        if net < self._config.min_profit_usd:
            return {
                "go": False,
                "reason": f"Net profit ${net:.2f} < min ${self._config.min_profit_usd:.2f}",
                "estimated_profit": estimated_profit,
                "total_cost": total_cost,
            }

        return {
            "go": True,
            "estimated_profit": estimated_profit,
            "total_cost": total_cost,
            "net_profit": net,
            "signal_strength": signal.signal_strength,
        }

    def execute_opportunity(
        self,
        signal: CryptoSignal,
        gas_cost_usd: float = 0.10,
    ) -> dict[str, Any]:
        """
        Execute a flash loan arbitrage opportunity.

        Returns execution result.
        """
        self._state.total_attempts += 1
        self._state.last_attempt_time = datetime.now(timezone.utc)

        # Use pluggable execution function
        if self._execute_fn:
            result = self._execute_fn(signal)
        else:
            result = {"success": False, "error": "No execution function configured"}

        if result.get("success"):
            profit = result.get("profit", 0.0)
            self._state.successful_attempts += 1
            self._state.total_profit_usd += profit
            self._state.total_gas_spent_usd += gas_cost_usd
            self._state.gas_spent_today_usd += gas_cost_usd
            self._state.last_success_time = datetime.now(timezone.utc)
            result["gas_cost"] = gas_cost_usd
        else:
            self._state.failed_attempts += 1
            self._state.failed_this_hour += 1
            self._state.total_gas_spent_usd += gas_cost_usd
            self._state.gas_spent_today_usd += gas_cost_usd

        return result

    def reset_daily(self) -> None:
        """Reset daily counters."""
        self._state.gas_spent_today_usd = 0.0
        self._state.is_paused = False
        self._state.pause_reason = ""

    def reset_hourly(self) -> None:
        """Reset hourly counters."""
        self._state.failed_this_hour = 0

    def summary(self) -> dict[str, Any]:
        """Get orchestrator summary."""
        return {
            "total_attempts": self._state.total_attempts,
            "success_rate": f"{self._state.success_rate:.1%}",
            "net_profit_usd": f"${self._state.net_profit:.2f}",
            "total_gas_usd": f"${self._state.total_gas_spent_usd:.2f}",
            "gas_today_usd": f"${self._state.gas_spent_today_usd:.2f}",
            "can_trade": self._state.can_trade,
        }
