"""
ENGINE 1: Burgess Statistical Arbitrage — Production Implementation.

v3 changes from v2:
    - Phase 2: Pure weighted composite score ranking, NO hard gates.
      All baskets scored, ranked, top_k carried forward.
    - Score weights: ADF + VR eigen2 projection heaviest (hypothesis),
      configurable via BurgessParams.score_weights for calibration.
    - Phase 4: Single-asset (laggard) trading instead of full spread.
      Signal from z-score of cointegration residual, execution on
      target asset only. Cuts transaction costs from 4 legs to 1.
    - Score decomposition saved per basket for post-hoc analysis of
      which score components predict backtest success.

Pipeline phases:
    Phase 1 — Full Universe Scan (stepwise regression, ADF, Hurst, HL)
    Phase 2 — Surface-Corrected Ranking (weighted composite, top_k)
    Phase 3 — Train/Test Split + OOS Validation
    Phase 4 — Walk-Forward Backtest (single-asset laggard trading)
    Phase 5 — Performance Metrics + Score Analysis

Surface: 4,352 precomputed grid points in data/surfaces.duckdb (480 MB).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from sklearn.linear_model import Ridge

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═════════════════════════════════════════════════════════════════════════════

# Default score weights — hypothesis: ADF + VR eigen2 are strongest predictors
DEFAULT_SCORE_WEIGHTS = {
    "adf_t":          0.25,
    "vr_eigen2_proj": 0.25,
    "hurst":          0.10,
    "half_life":      0.10,
    "vr_eigen1_proj": 0.10,
    "vr_eigen3_proj": 0.10,
    "vr_mahalanobis": 0.10,
}


@dataclass
class BurgessParams:
    """All tunable parameters for the full Burgess pipeline."""

    # ── Surface ──────────────────────────────────────────────────────
    surface_db_path: str | Path = "data/surfaces.duckdb"
    vr_max_lag: int = 50

    # ── Scan ─────────────────────────────────────────────────────────
    n_vars: int = 3
    ridge_alpha: float = 0.01

    # ── Scoring (no hard gates — pure weighted rank) ─────────────────
    score_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SCORE_WEIGHTS))

    # ── Train/Test ───────────────────────────────────────────────────
    train_frac: float = 0.70
    min_train_obs: int = 252
    min_test_obs: int = 63

    # ── OOS Validation ───────────────────────────────────────────────
    oos_adf_significance: float = 0.10

    # ── Walk-Forward ─────────────────────────────────────────────────
    wf_estimation_window: int = 504
    wf_signal_window: int = 63
    wf_step_size: int = 21

    # ── Signals ──────────────────────────────────────────────────────
    zscore_lookback: int = 63
    entry_threshold: float = 2.0
    exit_threshold: float = 0.5
    stop_loss_threshold: float = 4.0

    # ── Backtest ─────────────────────────────────────────────────────
    transaction_cost_bps: float = 10.0
    slippage_bps: float = 5.0
    initial_capital: float = 1_000_000.0
    max_position_pct: float = 0.10

    # ── Output ───────────────────────────────────────────────────────
    top_k: int = 100


@dataclass
class RegressionResult:
    target_idx: int
    basket_indices: list[int]
    betas: np.ndarray
    residuals: np.ndarray
    r_squared: float
    adj_r_squared: float


@dataclass
class StationarityResult:
    adf_t_value: float
    adf_p_value: float
    half_life: float
    hurst_exponent: float


@dataclass
class SurfaceTestResult:
    """Surface test output with per-component score decomposition."""
    stats: dict[str, dict[str, Any]]
    score_components: dict[str, float] = field(default_factory=dict)
    n_stats_available: int = 0


@dataclass
class CandidateBasket:
    target_idx: int
    basket_indices: list[int]
    regression: RegressionResult
    stationarity: StationarityResult
    surface_test: SurfaceTestResult | None = None
    composite_score: float = 0.0
    score_components: dict[str, float] = field(default_factory=dict)
    rank: int = 0


@dataclass
class OOSValidation:
    basket: CandidateBasket
    oos_regression: RegressionResult
    oos_stationarity: StationarityResult
    passed: bool


@dataclass
class TradeRecord:
    basket_idx: int
    entry_date_idx: int
    exit_date_idx: int
    entry_zscore: float
    exit_zscore: float
    direction: Literal["long", "short"]
    pnl: float
    pnl_pct: float
    hold_days: int
    cost: float


@dataclass
class BacktestResult:
    basket: CandidateBasket
    equity_curve: np.ndarray
    daily_returns: np.ndarray
    trades: list[TradeRecord]
    positions: np.ndarray
    zscore_series: np.ndarray


@dataclass
class PerformanceMetrics:
    basket_idx: int
    target_name: str
    basket_names: list[str]
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    n_trades: int
    avg_hold_days: float
    total_costs: float
    turnover: float
    # Score decomposition for analysis
    composite_score: float = 0.0
    score_components: dict[str, float] = field(default_factory=dict)


@dataclass
class BurgessOutput:
    all_regressions: list[RegressionResult]
    all_stationarity: list[StationarityResult]
    ranked_baskets: list[CandidateBasket]
    oos_results: list[OOSValidation]
    oos_passed_baskets: list[CandidateBasket]
    backtest_results: list[BacktestResult]
    performance: list[PerformanceMetrics]
    phase_times: dict[str, float]
    n_assets: int
    n_obs: int
    n_train: int
    n_test: int
    score_weights: dict[str, float] = field(default_factory=dict)


# ═════════════════════════════════════════════════════════════════════════════
# SURFACE REQUIREMENT
# ═════════════════════════════════════════════════════════════════════════════

def burgess_production_requirement():
    """MultiSurfaceRequirement matching the precomputed surface."""
    from praxis.stats.surface import MultiSurfaceRequirement, SurfaceAxis

    n_assets_small = sorted(set(list(range(3, 36)) + [40, 45, 50]))
    n_assets_large = sorted(set(
        list(range(55, 100, 5)) + [100] + list(range(150, 1001, 50))
    ))
    n_assets = n_assets_small + n_assets_large

    return MultiSurfaceRequirement(
        generator="stepwise_regression",
        universe_factory="random_walk",
        axes=[
            SurfaceAxis("n_assets", n_assets),
            SurfaceAxis("n_obs", list(range(200, 1001, 50))),
            SurfaceAxis("n_vars", [2, 3, 4, 5]),
        ],
        n_samples=500,
        seed=42,
        pct_conf=[10, 5, 1],
        scalar_extractors=["adf_t", "hurst", "half_life"],
        profile_collectors=["vr_profile"],
        profile_params={"vr_max_lag": 50},
    )


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 1: FULL UNIVERSE SCAN
# ═════════════════════════════════════════════════════════════════════════════

def corr2_coeff(A, B):
    """Row-wise correlation between two 2D arrays. Vectorized."""
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]
    ssA = (A_mA ** 2).sum(1)
    ssB = (B_mB ** 2).sum(1)
    denom = np.sqrt(np.dot(ssA[:, None], ssB[None]))
    denom = np.where(denom == 0, 1e-10, denom)
    return np.dot(A_mA, B_mB.T) / denom


def stepwise_regression(prices, target_idx, n_vars, alpha=0.01):
    """Greedy basket selection via correlation maximization with Ridge."""
    T, N = prices.shape
    y = prices[:, target_idx]
    current_residual = y.copy()

    mask = np.zeros(N, dtype=int)
    mask[target_idx] = 1
    basket_indices = []
    X_basket = np.empty((T, 0))

    for _ in range(min(n_vars, N - 1)):
        correlations = np.ma.array(
            np.abs(corr2_coeff(prices.T, current_residual[None, :])).ravel(),
            mask=mask,
        )
        if correlations.count() == 0:
            break
        best_idx = int(np.argmax(correlations))
        mask[best_idx] = 1
        basket_indices.append(best_idx)
        X_basket = np.column_stack([X_basket, prices[:, best_idx]])
        X_with_intercept = np.column_stack([np.ones(T), X_basket])
        clf = Ridge(alpha=alpha, fit_intercept=False)
        clf.fit(X_with_intercept, y)
        current_residual = y - clf.predict(X_with_intercept)

    if len(basket_indices) == 0:
        return RegressionResult(
            target_idx=target_idx, basket_indices=[],
            betas=np.array([y.mean()]), residuals=y - y.mean(),
            r_squared=0.0, adj_r_squared=0.0,
        )

    residuals = current_residual
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    p = len(basket_indices)
    adj_r2 = 1 - (1 - r2) * (T - 1) / (T - p - 1) if T > p + 1 else r2

    return RegressionResult(
        target_idx=target_idx, basket_indices=basket_indices,
        betas=clf.coef_, residuals=residuals,
        r_squared=r2, adj_r_squared=adj_r2,
    )


def compute_stationarity(residuals):
    """ADF + half-life + Hurst for a residual series."""
    from statsmodels.tsa.stattools import adfuller
    try:
        adf_result = adfuller(residuals, autolag="AIC")
        adf_t, adf_p = float(adf_result[0]), float(adf_result[1])
    except Exception:
        adf_t, adf_p = 0.0, 1.0
    return StationarityResult(
        adf_t_value=adf_t, adf_p_value=adf_p,
        half_life=_half_life(residuals),
        hurst_exponent=_hurst_rs(residuals),
    )


def _half_life(s):
    dy = np.diff(s)
    y_lag = s[:-1]
    X = np.column_stack([np.ones(len(dy)), y_lag])
    try:
        b, _, _, _ = np.linalg.lstsq(X, dy, rcond=None)
    except np.linalg.LinAlgError:
        return np.inf
    theta = -b[1]
    return np.log(2) / theta if theta > 1e-10 else np.inf


def _hurst_rs(s):
    n = len(s)
    if n < 40:
        return 0.5
    max_sz = n // 4
    min_sz = 10
    if max_sz <= min_sz:
        return 0.5
    sizes = np.unique(np.geomspace(min_sz, max_sz, min(20, max_sz - min_sz)).astype(int))
    log_sizes, log_rs = [], []
    for sz in sizes:
        n_chunks = n // sz
        if n_chunks < 2:
            continue
        rs_values = []
        for i in range(n_chunks):
            chunk = s[i * sz : (i + 1) * sz]
            centered = chunk - chunk.mean()
            cumdev = np.cumsum(centered)
            R = cumdev.max() - cumdev.min()
            S = chunk.std(ddof=1)
            if S > 1e-10:
                rs_values.append(R / S)
        if rs_values:
            log_sizes.append(np.log(sz))
            log_rs.append(np.log(np.mean(rs_values)))
    if len(log_sizes) < 3:
        return 0.5
    from scipy import stats as sp_stats
    slope, _, _, _, _ = sp_stats.linregress(log_sizes, log_rs)
    return float(np.clip(slope, 0.0, 1.0))


def full_universe_scan(prices, params, progress_callback=None):
    """Phase 1: Stepwise regression on every asset as target."""
    T, N = prices.shape
    regressions = []
    stationarity_results = []
    for i in range(N):
        reg = stepwise_regression(prices, i, params.n_vars, params.ridge_alpha)
        regressions.append(reg)
        stat = compute_stationarity(reg.residuals)
        stationarity_results.append(stat)
        if progress_callback and (i + 1) % 50 == 0:
            progress_callback(i + 1, N)
    return regressions, stationarity_results


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 2: SURFACE-CORRECTED RANKING (no hard gates)
# ═════════════════════════════════════════════════════════════════════════════

def surface_rank(regressions, stationarity_results, surface, req,
                 n_assets, n_obs, params, progress_callback=None):
    """
    Score and rank ALL baskets — no gates. Pure weighted composite.
    Score decomposition saved per basket for calibration analysis.
    """
    weights = params.score_weights
    candidates = []

    for i, (reg, stat) in enumerate(zip(regressions, stationarity_results)):
        if len(reg.basket_indices) == 0:
            continue

        n_vars = len(reg.basket_indices)
        try:
            test_result = surface.test_candidate(
                req, reg.residuals,
                n_assets=n_assets, n_obs=n_obs, n_vars=n_vars,
            )
        except Exception as e:
            logger.warning("Surface lookup failed for target %d: %s", reg.target_idx, e)
            continue

        components, total = _score_basket(test_result, weights)

        surface_test = SurfaceTestResult(
            stats=test_result, score_components=components,
            n_stats_available=len(test_result),
        )

        candidate = CandidateBasket(
            target_idx=reg.target_idx, basket_indices=reg.basket_indices,
            regression=reg, stationarity=stat,
            surface_test=surface_test,
            composite_score=total, score_components=components,
        )
        candidates.append(candidate)

        if progress_callback and (i + 1) % 100 == 0:
            progress_callback(i + 1, len(regressions))

    candidates.sort(key=lambda c: c.composite_score, reverse=True)
    for i, c in enumerate(candidates):
        c.rank = i + 1

    return candidates


def _score_basket(test_result, weights):
    """
    Per-component and total composite score.

    For each stat: how far observed exceeds the 5% corrected CV,
    normalized by CV magnitude. Negative scores allowed (penalizes
    baskets that don't even reach the null).

    Direction conventions:
        ADF t-stat:    more negative = better  (lower is better)
        Hurst:         lower = better (< 0.5 = mean-reverting)
        Half-life:     lower = better (faster reversion)
        VR eigen*_proj: more extreme = better (magnitude matters)
        VR mahalanobis: higher = better (further from null)
    """
    LOWER_IS_BETTER = {"adf_t", "hurst", "half_life"}
    HIGHER_IS_BETTER = {"vr_mahalanobis"}
    MAGNITUDE_MATTERS = {"vr_eigen1_proj", "vr_eigen2_proj", "vr_eigen3_proj"}

    ref_pct = 5  # use 5% CV as scoring reference

    components = {}
    total = 0.0

    for stat_name, weight in weights.items():
        if stat_name not in test_result:
            components[stat_name] = 0.0
            continue

        r = test_result[stat_name]
        value = r.get("value", 0)
        cvs = r.get("critical_values", {})

        if ref_pct not in cvs:
            components[stat_name] = 0.0
            continue

        cv = cvs[ref_pct]
        denom = max(abs(cv), 1e-10)

        if stat_name in LOWER_IS_BETTER:
            excess = (cv - value) / denom
        elif stat_name in HIGHER_IS_BETTER:
            excess = (value - cv) / denom
        elif stat_name in MAGNITUDE_MATTERS:
            excess = (abs(value) - abs(cv)) / denom
        else:
            excess = (value - cv) / denom

        component_score = weight * excess
        components[stat_name] = component_score
        total += component_score

    return components, total


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 3: TRAIN/TEST SPLIT + OOS VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def split_train_test(prices, train_frac, min_train=252, min_test=63):
    T = prices.shape[0]
    split = max(min_train, min(T - min_test, int(T * train_frac)))
    return prices[:split], prices[split:], split


def validate_oos(test_prices, basket, params):
    """Re-estimate on test data, check if cointegration persists."""
    T = test_prices.shape[0]
    y = test_prices[:, basket.target_idx]
    if not basket.basket_indices:
        dummy_reg = RegressionResult(basket.target_idx, [], np.array([0]), y, 0, 0)
        dummy_stat = StationarityResult(0, 1.0, np.inf, 0.5)
        return OOSValidation(basket, dummy_reg, dummy_stat, False)
    X = np.column_stack([np.ones(T), test_prices[:, basket.basket_indices]])
    clf = Ridge(alpha=params.ridge_alpha, fit_intercept=False)
    clf.fit(X, y)
    residuals = y - clf.predict(X)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    p = len(basket.basket_indices)
    adj_r2 = 1 - (1 - r2) * (T - 1) / (T - p - 1) if T > p + 1 else r2
    oos_reg = RegressionResult(
        target_idx=basket.target_idx, basket_indices=basket.basket_indices,
        betas=clf.coef_, residuals=residuals, r_squared=r2, adj_r_squared=adj_r2,
    )
    oos_stat = compute_stationarity(residuals)
    passed = oos_stat.adf_p_value < params.oos_adf_significance
    return OOSValidation(basket=basket, oos_regression=oos_reg,
                         oos_stationarity=oos_stat, passed=passed)


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 4: WALK-FORWARD BACKTEST — SINGLE-ASSET (LAGGARD) TRADING
# ═════════════════════════════════════════════════════════════════════════════

def walk_forward_windows(T, estimation_window, signal_window, step_size):
    windows = []
    est_start = 0
    while est_start + estimation_window + signal_window <= T:
        est_end = est_start + estimation_window
        sig_start = est_end
        sig_end = min(sig_start + signal_window, T)
        windows.append((est_start, est_end, sig_start, sig_end))
        est_start += step_size
    return windows


def walk_forward_single_basket(prices, target_idx, basket_indices, params):
    """
    Walk-forward backtest: SINGLE-ASSET (laggard) trading.

    Signal from z-score of cointegration residual. Execution on
    TARGET asset ONLY — the one that deviated from basket-implied
    fair value. 1 leg, not 4.

        z > entry  -> target expensive -> SHORT target
        z < -entry -> target cheap     -> LONG target

    P&L from target asset price returns. Costs for 1 leg round-trip.
    """
    T = prices.shape[0]
    windows = walk_forward_windows(
        T, params.wf_estimation_window, params.wf_signal_window, params.wf_step_size,
    )
    if not windows:
        return np.zeros(0), np.zeros(0), []

    all_pnl = np.zeros(T)
    all_positions = np.zeros(T)
    all_trades = []

    target_prices = prices[:, target_idx]
    target_returns = np.zeros(T)
    target_returns[1:] = np.diff(target_prices) / target_prices[:-1]

    for est_start, est_end, sig_start, sig_end in windows:
        # Estimation: fit betas
        est_prices = prices[est_start:est_end]
        y_est = est_prices[:, target_idx]
        X_est = np.column_stack([np.ones(est_end - est_start), est_prices[:, basket_indices]])
        clf = Ridge(alpha=params.ridge_alpha, fit_intercept=False)
        clf.fit(X_est, y_est)
        est_residuals = y_est - clf.predict(X_est)

        # Signal period
        sig_prices = prices[sig_start:sig_end]
        y_sig = sig_prices[:, target_idx]
        X_sig = np.column_stack([np.ones(sig_end - sig_start), sig_prices[:, basket_indices]])
        sig_residuals = y_sig - clf.predict(X_sig)

        combined = np.concatenate([est_residuals[-params.zscore_lookback:], sig_residuals])
        lb = params.zscore_lookback
        zscores = np.zeros(len(sig_residuals))
        for i in range(len(sig_residuals)):
            window = combined[i : i + lb]
            mu, sigma = window.mean(), window.std()
            if sigma > 1e-10:
                zscores[i] = (combined[i + lb] - mu) / sigma

        # Execute: trade TARGET ONLY
        position = 0
        entry_idx = -1
        entry_z = 0.0
        entry_price = 0.0

        for i in range(len(zscores)):
            z = zscores[i]
            global_idx = sig_start + i

            if position == 0:
                if z >= params.entry_threshold:
                    position = -1  # short target (expensive)
                    entry_idx = global_idx
                    entry_z = z
                    entry_price = target_prices[global_idx]
                elif z <= -params.entry_threshold:
                    position = 1   # long target (cheap)
                    entry_idx = global_idx
                    entry_z = z
                    entry_price = target_prices[global_idx]
            else:
                should_exit = False
                if position == 1 and z >= -params.exit_threshold:
                    should_exit = True
                elif position == -1 and z <= params.exit_threshold:
                    should_exit = True
                elif abs(z) >= params.stop_loss_threshold:
                    should_exit = True

                if should_exit:
                    hold_days = global_idx - entry_idx
                    if hold_days > 0 and entry_idx >= 0:
                        exit_price = target_prices[global_idx]
                        asset_return = (exit_price - entry_price) / entry_price
                        # 1 leg round-trip cost
                        cost_frac = (params.transaction_cost_bps + params.slippage_bps) * 2 / 10_000
                        pnl_pct = position * asset_return - cost_frac
                        pnl = pnl_pct * entry_price

                        all_trades.append(TradeRecord(
                            basket_idx=target_idx,
                            entry_date_idx=entry_idx, exit_date_idx=global_idx,
                            entry_zscore=entry_z, exit_zscore=z,
                            direction="long" if position == 1 else "short",
                            pnl=pnl, pnl_pct=pnl_pct,
                            hold_days=hold_days, cost=cost_frac * entry_price,
                        ))
                    position = 0
                    entry_idx = -1

            all_positions[global_idx] = position
            if global_idx > 0 and position != 0:
                all_pnl[global_idx] = position * target_returns[global_idx]

    return all_pnl, all_positions, all_trades


def backtest_basket(prices, basket, params):
    """Full backtest via walk-forward, single-asset trading."""
    daily_pnl, positions, trades = walk_forward_single_basket(
        prices, basket.target_idx, basket.basket_indices, params,
    )
    allocation = params.initial_capital * params.max_position_pct
    daily_returns = daily_pnl  # already in return space
    equity_curve = allocation * (1 + np.cumsum(daily_returns))

    T = prices.shape[0]
    y = prices[:, basket.target_idx]
    X = np.column_stack([np.ones(T), prices[:, basket.basket_indices]])
    clf = Ridge(alpha=params.ridge_alpha, fit_intercept=False)
    clf.fit(X, y)
    full_residuals = y - clf.predict(X)
    zscores = np.zeros(T)
    lb = params.zscore_lookback
    for i in range(lb, T):
        w = full_residuals[i - lb : i]
        mu, sigma = w.mean(), w.std()
        if sigma > 1e-10:
            zscores[i] = (full_residuals[i] - mu) / sigma

    return BacktestResult(
        basket=basket, equity_curve=equity_curve,
        daily_returns=daily_returns, trades=trades,
        positions=positions, zscore_series=zscores,
    )


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 5: PERFORMANCE METRICS
# ═════════════════════════════════════════════════════════════════════════════

def compute_metrics(bt, asset_names, trading_days_per_year=252):
    rets = bt.daily_returns
    trades = bt.trades
    total_ret = np.sum(rets)
    n_days = max(len(rets), 1)
    years = n_days / trading_days_per_year
    annual_return = total_ret / years if years > 0 else 0.0
    annual_vol = np.std(rets) * np.sqrt(trading_days_per_year) if len(rets) > 1 else 1e-10
    sharpe = annual_return / annual_vol if annual_vol > 1e-10 else 0.0
    neg_rets = rets[rets < 0]
    downside_std = np.std(neg_rets) * np.sqrt(trading_days_per_year) if len(neg_rets) > 1 else 1e-10
    sortino = annual_return / downside_std if downside_std > 1e-10 else 0.0
    cum = np.cumsum(rets)
    running_max = np.maximum.accumulate(cum)
    drawdowns = cum - running_max
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
    dd_duration = 0
    max_dd_duration = 0
    for i in range(len(drawdowns)):
        if drawdowns[i] < 0:
            dd_duration += 1
            max_dd_duration = max(max_dd_duration, dd_duration)
        else:
            dd_duration = 0
    calmar = annual_return / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0
    n_trades = len(trades)
    if n_trades > 0:
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        win_rate = len(wins) / n_trades
        avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0.0
        avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0.0
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        avg_hold = np.mean([t.hold_days for t in trades])
        total_costs = sum(t.cost for t in trades)
    else:
        win_rate = avg_win = avg_loss = profit_factor = avg_hold = total_costs = 0.0
    turnover = (2 * n_trades) / years if years > 0 else 0.0
    target_name = asset_names[bt.basket.target_idx] if bt.basket.target_idx < len(asset_names) else f"A{bt.basket.target_idx}"
    basket_names = [asset_names[i] if i < len(asset_names) else f"A{i}" for i in bt.basket.basket_indices]

    return PerformanceMetrics(
        basket_idx=bt.basket.target_idx, target_name=target_name,
        basket_names=basket_names, annual_return=annual_return,
        annual_volatility=annual_vol, sharpe_ratio=sharpe,
        sortino_ratio=sortino, calmar_ratio=calmar,
        max_drawdown=max_dd, max_drawdown_duration=max_dd_duration,
        win_rate=win_rate, avg_win=float(avg_win), avg_loss=float(avg_loss),
        profit_factor=float(profit_factor), n_trades=n_trades,
        avg_hold_days=float(avg_hold), total_costs=float(total_costs),
        turnover=turnover, composite_score=bt.basket.composite_score,
        score_components=dict(bt.basket.score_components),
    )


# ═════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

class BurgessEngine:
    """
    Full Burgess pipeline: surface-corrected ranking + single-asset
    laggard trading. No hard gates — pure weighted composite score.
    """

    def run(self, prices, asset_names, params=None, progress_callback=None):
        if params is None:
            params = BurgessParams()
        T, N = prices.shape
        phase_times = {}

        def _log(phase, msg):
            if progress_callback:
                progress_callback(phase, msg)
            logger.info("[%s] %s", phase, msg)

        from praxis.stats.surface import CompositeSurface
        surface = CompositeSurface(db_path=params.surface_db_path)
        req = burgess_production_requirement()
        _log("INIT", f"Surface connected: {params.surface_db_path}")
        _log("INIT", f"Score weights: {params.score_weights}")

        train, test, split_idx = split_train_test(
            prices, params.train_frac, params.min_train_obs, params.min_test_obs
        )
        T_train, T_test = train.shape[0], test.shape[0]
        _log("SPLIT", f"Train: {T_train} obs, Test: {T_test} obs (split at {split_idx})")

        # Phase 1
        _log("PHASE1", f"Scanning {N} assets x {params.n_vars} vars...")
        t0 = time.perf_counter()
        def _scan_progress(i, n):
            _log("PHASE1", f"  {i}/{n} targets scanned")
        regressions, stationarity = full_universe_scan(train, params, _scan_progress)
        phase_times["scan"] = time.perf_counter() - t0
        n_adf_raw = sum(1 for s in stationarity if s.adf_p_value < 0.05)
        _log("PHASE1", f"  Done: {N} scanned, {n_adf_raw} raw ADF sig (p<0.05)")

        # Phase 2 — pure ranking
        _log("PHASE2", f"Scoring {N} baskets (pure rank, backtest ALL)...")
        t0 = time.perf_counter()
        def _rank_progress(i, n):
            _log("PHASE2", f"  {i}/{n} scored")
        ranked = surface_rank(
            regressions, stationarity, surface, req,
            n_assets=N, n_obs=T_train, params=params, progress_callback=_rank_progress,
        )
        phase_times["surface_rank"] = time.perf_counter() - t0
        _log("PHASE2", f"  {len(ranked)} scored, backtesting all")
        if ranked:
            _log("PHASE2", f"  Score range: {ranked[0].composite_score:.4f} -> {ranked[-1].composite_score:.4f}")

        # Phase 3 — OOS
        # Phase 3 — OOS all ranked baskets
        _log("PHASE3", f"Validating {len(ranked)} baskets OOS...")
        t0 = time.perf_counter()
        oos_results = [validate_oos(test, b, params) for b in ranked]
        oos_passed = [r for r in oos_results if r.passed]
        oos_baskets = [r.basket for r in oos_passed]
        phase_times["oos_validation"] = time.perf_counter() - t0
        _log("PHASE3", f"  {len(oos_passed)}/{len(ranked)} passed OOS")

        # Phase 4 — single-asset backtest
        _log("PHASE4", f"Backtesting {len(oos_baskets)} baskets (single-asset laggard)...")
        t0 = time.perf_counter()
        backtest_results = []
        for i, basket in enumerate(oos_baskets):
            bt = backtest_basket(prices, basket, params)
            backtest_results.append(bt)
            if (i + 1) % 10 == 0:
                _log("PHASE4", f"  {i+1}/{len(oos_baskets)} backtested")
        phase_times["backtest"] = time.perf_counter() - t0

        # Phase 5
        _log("PHASE5", "Computing performance metrics...")
        t0 = time.perf_counter()
        performance = [compute_metrics(bt, asset_names) for bt in backtest_results]
        performance.sort(key=lambda p: p.sharpe_ratio, reverse=True)
        phase_times["metrics"] = time.perf_counter() - t0

        total_time = sum(phase_times.values())
        _log("DONE", f"Pipeline complete in {total_time:.1f}s")
        _log("DONE", f"  Scanned: {N} | Ranked: {len(ranked)} | OOS pass: {len(oos_passed)} | Backtested: {len(backtest_results)}")
        if performance:
            best = performance[0]
            _log("DONE", f"  Best Sharpe: {best.sharpe_ratio:.3f} ({best.target_name} ~ {best.basket_names})")

        return BurgessOutput(
            all_regressions=regressions, all_stationarity=stationarity,
            ranked_baskets=ranked, oos_results=oos_results,
            oos_passed_baskets=oos_baskets, backtest_results=backtest_results,
            performance=performance, phase_times=phase_times,
            n_assets=N, n_obs=T, n_train=T_train, n_test=T_test,
            score_weights=dict(params.score_weights),
        )
