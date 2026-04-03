"""
CPO Core — Generic Conditional Portfolio Optimization Engine.

Strategy-agnostic framework for:
    Phase 2 — Parameter grid search (strategy provides execution)
    Phase 3 — Random Forest: (daily_features, config_params) → P(profitable)
    Phase 4 — Kelly Vector portfolio of RF-selected models

Any trading strategy (pairs, TA, crypto, etc.) implements the
TradingStrategy protocol and plugs into this framework.
"""
from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import os
os.environ["PYTHONWARNINGS"] = "ignore"

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# STRATEGY PROTOCOL — implement this for any new trading strategy
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelSpec:
    """Base class for a tradeable model (pair, TA signal, etc.)."""
    model_id: str


@dataclass
class ConfigSpec:
    """Base class for a parameter configuration."""
    config_id: int

    def to_feature_vector(self) -> list[float]:
        """Return normalized parameter values as RF features."""
        raise NotImplementedError

    @staticmethod
    def param_names() -> list[str]:
        """Return names of config parameters (for feature importance)."""
        raise NotImplementedError


@runtime_checkable
class TradingStrategy(Protocol):
    """
    Protocol that any trading strategy must implement.

    The CPO framework calls these methods without knowing anything
    about the specific strategy (pairs, TA, crypto, etc.).
    """

    def get_models(self) -> list[ModelSpec]:
        """Return the universe of tradeable models."""
        ...

    def get_param_grid(self) -> list[ConfigSpec]:
        """Return parameter configurations to sweep."""
        ...

    def daily_feature_names(self) -> list[str]:
        """Return names of daily lagged features."""
        ...

    def config_param_names(self) -> list[str]:
        """Return names of config parameters (for RF feature tracking)."""
        ...

    def config_to_features(self, config: ConfigSpec) -> list[float]:
        """Convert a config to normalized feature values for RF input."""
        ...

    def compute_features(self, model: ModelSpec, as_of_date: str,
                         data: dict) -> np.ndarray | None:
        """
        Compute lagged daily features for a model as of a given date.
        Returns feature vector or None if insufficient data.
        Must use data STRICTLY up to (not beyond) as_of_date.
        """
        ...

    def run_single_day(self, model: ModelSpec, config: ConfigSpec,
                       day: Any, data: dict) -> dict:
        """
        Execute one model+config for one day.
        Returns: {daily_return, gross_return, n_trades, ...}
        """
        ...

    def run_model_year(self, model: ModelSpec, data: dict,
                       param_grid: list[ConfigSpec]) -> pd.DataFrame:
        """
        Run all configs for all days for one model (Phase 2).
        Returns DataFrame with columns:
            model_id, date, config_id, daily_return, gross_return, n_trades
        """
        ...

    def fetch_training_data(self, models: list[ModelSpec],
                            start: str, end: str) -> dict:
        """Fetch all data needed for Phase 2 training period."""
        ...

    def fetch_oos_data(self, models: list[ModelSpec],
                       start: str, end: str) -> dict:
        """Fetch all data needed for Phase 4 OOS period."""
        ...

    def fetch_warmup_daily(self, models: list[ModelSpec],
                           start: str, end: str) -> dict[str, pd.Series]:
        """Fetch daily prices for feature warmup period."""
        ...

    def get_daily_prices(self, data: dict,
                         models: list[ModelSpec]) -> dict[str, pd.Series]:
        """Extract daily close prices from data (for Kelly covariance)."""
        ...

    def get_trading_days(self, data: dict) -> list:
        """Return valid trading days from OOS data."""
        ...

    def prepare_warmup(self, daily_prices: dict,
                       warmup_daily: dict) -> dict:
        """Prepend warmup daily prices to OOS daily prices."""
        ...


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 3: RANDOM FOREST — CONDITIONAL PROFITABILITY CLASSIFIER
# ═════════════════════════════════════════════════════════════════════════════

def train_conditional_model(features_df: pd.DataFrame,
                            returns_df: pd.DataFrame,
                            model_id: str,
                            param_grid: list[ConfigSpec],
                            daily_feature_names: list[str],
                            config_param_names: list[str],
                            n_estimators: int = 200,
                            max_depth: int = 5,
                            ) -> dict:
    """
    Train a Random Forest CLASSIFIER for one model.

    Each training sample = (daily_features, config_params) → profitable? (1/0)
    This gives ~N_configs × N_days samples per model.

    The RF learns the INTERACTION between market conditions and parameter
    choice. At prediction time: sweep all configs, pick the one with
    highest P(profitable).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from collections import Counter

    model_features = features_df[features_df["model_id"] == model_id].copy()
    model_returns = returns_df[returns_df["model_id"] == model_id].copy()

    if model_features.empty or model_returns.empty:
        return {"model": None, "error": "no data"}

    # Build config lookup: config_id → normalized feature vector
    config_lookup = {c.config_id: c.to_feature_vector() for c in param_grid}

    # Daily features
    feature_cols = [f for f in daily_feature_names if f in features_df.columns]
    if not feature_cols:
        return {"model": None, "error": "no matching feature columns"}

    model_features = model_features.set_index("date")
    trading_dates = sorted(model_returns["date"].unique())
    feature_dates = set(model_features.index)

    # Build training matrix: [daily_features..., config_params...] → profitable
    X_rows = []
    y_rows = []

    for i in range(len(trading_dates) - 1):
        feat_date = trading_dates[i]
        trade_date = trading_dates[i + 1]

        if feat_date not in feature_dates:
            continue

        daily_feats = model_features.loc[feat_date, feature_cols]
        if hasattr(daily_feats, 'values'):
            daily_vec = daily_feats.values.astype(float)
        else:
            daily_vec = np.array([float(daily_feats)])

        if not np.all(np.isfinite(daily_vec)):
            continue

        # All configs on the trade date
        day_returns = model_returns[model_returns["date"] == trade_date]
        for _, row in day_returns.iterrows():
            cid = row["config_id"]
            gross_ret = row.get("gross_return", row["daily_return"])
            if not np.isfinite(gross_ret) or cid not in config_lookup:
                continue
            config_vec = config_lookup[cid]
            X_rows.append(np.concatenate([daily_vec, config_vec]))
            y_rows.append(1 if gross_ret > 0 else 0)

    X = np.array(X_rows)
    y = np.array(y_rows)

    # Clean NaN/Inf
    valid = np.all(np.isfinite(X), axis=1)
    X, y = X[valid], y[valid]

    if len(X) < 100:
        return {"model": None, "error": f"only {len(X)} valid samples (need 100+)"}

    n_pos = int(y.sum())
    base_rate = n_pos / len(y)

    all_feature_names = feature_cols + [f"cfg_{n}" for n in config_param_names]

    # Train RF
    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=20, min_samples_split=40,
        class_weight="balanced",
        random_state=42, n_jobs=1,
    )
    rf.fit(X, y)

    proba_train = rf.predict_proba(X)[:, 1]
    try:
        auc = roc_auc_score(y, proba_train)
    except ValueError:
        auc = 0.5

    # Mean return by outcome (for Kelly expected return sizing)
    ret_vals = model_returns["daily_return"].dropna().values
    if len(ret_vals) > 0:
        mean_win = float(np.mean(ret_vals[ret_vals > 0])) if (ret_vals > 0).any() else 0.0
        mean_loss = float(np.mean(ret_vals[ret_vals <= 0])) if (ret_vals <= 0).any() else 0.0
    else:
        mean_win, mean_loss = 0.01, -0.01

    importance = dict(zip(all_feature_names, rf.feature_importances_))

    return {
        "model": rf,
        "feature_cols": feature_cols,
        "feature_importance": importance,
        "train_score": auc,
        "base_rate": base_rate,
        "mean_win": mean_win,
        "mean_loss": mean_loss,
        "n_samples": len(X),
        "n_days": len(trading_dates),
    }


def predict_model(trained_model: dict, features: np.ndarray,
                   param_grid: list[ConfigSpec]
                   ) -> tuple[ConfigSpec, float, float]:
    """
    Predict P(profitable) for all configs, pick the best.

    Returns: (best_config, p_profitable, expected_return)
    """
    model = trained_model.get("model")
    if model is None:
        return param_grid[0], 0.0, 0.0

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Map to trained features if needed
    feature_cols = trained_model.get("feature_cols")
    if feature_cols and len(features) > len(feature_cols):
        # Caller must ensure feature order matches
        features = features[:len(feature_cols)]

    # Build prediction matrix: one row per config
    X_pred = np.array([
        np.concatenate([features, c.to_feature_vector()])
        for c in param_grid
    ])
    X_pred = np.nan_to_num(X_pred, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        probas = model.predict_proba(X_pred)[:, 1]
    except Exception:
        return param_grid[0], 0.5, 0.0

    best_idx = int(np.argmax(probas))
    best_config = param_grid[best_idx]
    p_profitable = float(probas[best_idx])

    mean_win = trained_model.get("mean_win", 0.01)
    mean_loss = trained_model.get("mean_loss", -0.01)
    expected_return = p_profitable * mean_win + (1 - p_profitable) * mean_loss

    return best_config, p_profitable, float(expected_return)


# ═════════════════════════════════════════════════════════════════════════════
# MODEL PRE-FILTER — Remove structurally unprofitable model types
# ═════════════════════════════════════════════════════════════════════════════

def filter_models_by_training(returns_df: pd.DataFrame,
                               group_fn=None,
                               min_group_mean: float = 0.0,
                               ) -> tuple[list[str], dict]:
    """
    Filter models based on training data profitability.

    Uses ONLY in-sample (training) data to determine which models
    or model groups have structural edge. No OOS contamination.

    group_fn: maps model_id → group key (e.g. "BTC_MACD" → "MACD").
        If None, each model is its own group.
    min_group_mean: minimum average gross return to keep a group.
        Default 0.0 = must be positive on average.

    Returns: (kept_model_ids, group_stats_dict)
    """
    if "gross_return" not in returns_df.columns:
        # Fall back to daily_return if gross not available
        ret_col = "daily_return"
    else:
        ret_col = "gross_return"

    # Compute per-model average return
    model_means = returns_df.groupby("model_id")[ret_col].mean()

    if group_fn is None:
        # Filter at individual model level
        kept = [mid for mid, mean_ret in model_means.items()
                if mean_ret > min_group_mean]
        stats = {mid: {"mean_return": float(mean_ret), "kept": mean_ret > min_group_mean}
                 for mid, mean_ret in model_means.items()}
    else:
        # Group models and filter at group level
        model_ids = returns_df["model_id"].unique()
        groups = {}
        for mid in model_ids:
            gkey = group_fn(mid)
            groups.setdefault(gkey, []).append(mid)

        # Compute group-level mean (pool all models in group)
        group_means = {}
        for gkey, mids in groups.items():
            group_rets = returns_df[returns_df["model_id"].isin(mids)][ret_col]
            group_means[gkey] = float(group_rets.mean())

        # Keep groups above threshold
        kept_groups = {gk for gk, gm in group_means.items() if gm > min_group_mean}
        kept = [mid for mid in model_ids if group_fn(mid) in kept_groups]

        stats = {}
        for gkey in sorted(group_means.keys(), key=lambda k: -group_means[k]):
            is_kept = gkey in kept_groups
            n_models = len(groups[gkey])
            stats[gkey] = {
                "mean_return": group_means[gkey],
                "n_models": n_models,
                "kept": is_kept,
            }

    return kept, stats


# ═════════════════════════════════════════════════════════════════════════════
# KELLY VECTOR PORTFOLIO OPTIMIZATION
# ═════════════════════════════════════════════════════════════════════════════

def _ledoit_wolf_shrink(cov: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrinkage toward scaled identity."""
    n = cov.shape[0]
    if n <= 1:
        return cov + 1e-6 * np.eye(n)
    trace = np.trace(cov)
    mu_target = trace / n
    target = mu_target * np.eye(n)
    delta = cov - target
    delta_sq_sum = np.sum(delta ** 2)
    if delta_sq_sum < 1e-12:
        return cov + 1e-6 * np.eye(n)
    alpha = np.clip(n / (n + delta_sq_sum / (trace ** 2 / n)), 0.1, 0.9)
    return alpha * target + (1 - alpha) * cov + 1e-6 * np.eye(n)


def kelly_vector(expected_returns: np.ndarray,
                 cov_matrix: np.ndarray,
                 max_leverage: float = 2.0,
                 kelly_fraction: float = 0.5) -> np.ndarray:
    """Half-Kelly vector with Ledoit-Wolf shrinkage."""
    n = len(expected_returns)
    if n == 0:
        return np.array([])
    mu = np.maximum(expected_returns, 0)
    if mu.sum() < 1e-10:
        return np.zeros(n)
    cov = _ledoit_wolf_shrink(cov_matrix)
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)
    weights = kelly_fraction * (cov_inv @ mu)
    weights = np.maximum(weights, 0)
    total = weights.sum()
    if total > max_leverage:
        weights *= (max_leverage / total)
    return weights


def compute_allocation(model_predictions: list[dict],
                       returns_history: pd.DataFrame | None = None,
                       max_leverage: float = 2.0,
                       max_weight_per_model: float = 0.05,
                       prob_threshold: float = 0.50,
                       min_lift: float = 0.0,
                       mode: str = "equal_weight",
                       corr_threshold: float = 0.85,
                       lookback_days: int = 60,
                       ) -> dict[str, float]:
    """
    Portfolio allocation — configurable per strategy.

    Gate logic:
        If min_lift > 0: P > model's base_rate + min_lift
            (RF must find a config meaningfully better than average)
        Else: P > prob_threshold (absolute gate)

    Modes:
        "equal_weight" — best for short-lived models
        "kelly" — best for persistent models with return history
    """
    # Gate: lift-based or absolute
    if min_lift > 0:
        candidates = [p for p in model_predictions
                      if p.get("p_profitable", 0) > p.get("base_rate", 0.5) + min_lift]
    else:
        candidates = [p for p in model_predictions
                      if p.get("p_profitable", 0) > prob_threshold]

    if not candidates:
        return {}

    model_ids = [p["model_id"] for p in candidates]
    probs = np.array([p["p_profitable"] for p in candidates])
    mu = np.array([p["expected_return"] for p in candidates])
    n = len(candidates)

    # ── Mode: equal_weight ────────────────────────────────────────
    if mode == "equal_weight":
        weight = min(max_leverage / n, max_weight_per_model)
        return {p["model_id"]: float(weight) for p in candidates}

    # ── Mode: kelly ───────────────────────────────────────────────
    # Stage 2: Correlation dedup (only with sufficient history)
    if (returns_history is not None and not returns_history.empty
            and "date" in returns_history.columns
            and returns_history["date"].nunique() >= 10):

        recent = returns_history[returns_history["date"].isin(
            returns_history["date"].unique()[-lookback_days:]
        )]
        pivot = recent.pivot_table(index="date", columns="model_id",
                                   values="daily_return", aggfunc="first")

        available_ids = [m for m in model_ids if m in pivot.columns]
        if len(available_ids) >= 2:
            corr_matrix = pivot[available_ids].corr().values
            sorted_idx = np.argsort(-probs[:len(available_ids)])
            keep_mask = np.ones(len(available_ids), dtype=bool)

            for i in range(len(sorted_idx)):
                idx_i = sorted_idx[i]
                if not keep_mask[idx_i]:
                    continue
                for j in range(i + 1, len(sorted_idx)):
                    idx_j = sorted_idx[j]
                    if not keep_mask[idx_j]:
                        continue
                    if (idx_i < corr_matrix.shape[0]
                            and idx_j < corr_matrix.shape[1]
                            and abs(corr_matrix[idx_i, idx_j]) > corr_threshold):
                        keep_mask[idx_j] = False

            survived_ids = [available_ids[i] for i in range(len(available_ids))
                           if keep_mask[i]]
            candidates = [p for p in candidates if p["model_id"] in survived_ids]
            model_ids = [p["model_id"] for p in candidates]
            mu = np.array([p["expected_return"] for p in candidates])
            n = len(candidates)

    if not candidates:
        return {}

    # Stage 3: Kelly vector allocation
    # Need return history for covariance — fall back to equal weight if unavailable
    if (returns_history is None or returns_history.empty
            or "date" not in returns_history.columns
            or returns_history["date"].nunique() < 10):
        weight = min(max_leverage / n, max_weight_per_model)
        return {mid: float(weight) for mid in model_ids}

    recent = returns_history[returns_history["date"].isin(
        returns_history["date"].unique()[-lookback_days:]
    )]
    pivot = recent.pivot_table(index="date", columns="model_id",
                               values="daily_return", aggfunc="first")

    available = [m for m in model_ids if m in pivot.columns]
    if len(available) < 1:
        return {}

    mu_avail = np.array([mu[model_ids.index(m)] for m in available])
    cov_matrix = pivot[available].cov().values

    if cov_matrix.shape[0] != len(available):
        return {}

    weights = kelly_vector(mu_avail, cov_matrix, max_leverage)
    weights = np.minimum(weights, max_weight_per_model)

    allocation = {}
    for mid, w in zip(available, weights):
        if w > 0.0001:
            allocation[mid] = float(w)

    return allocation


# ═════════════════════════════════════════════════════════════════════════════
# PHASE ORCHESTRATION — Generic pipeline that works with any strategy
# ═════════════════════════════════════════════════════════════════════════════

def run_phase2(strategy: TradingStrategy, output_dir: Path,
               features_suffix: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Phase 2: Run all configs on all models for training period.
    Strategy provides execution; this handles orchestration and caching.

    features_suffix: appended to features cache filename to allow
        multiple feature sets (e.g. "funding", "funding+regime", "regime")
        to coexist without overwriting each other. Returns cache is shared.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    returns_path = output_dir / "phase2_returns.parquet"
    sfx = f"_{features_suffix}" if features_suffix else ""
    features_path = output_dir / f"phase2_features{sfx}.parquet"

    models = strategy.get_models()
    param_grid = strategy.get_param_grid()
    data = strategy.fetch_training_data(models, "", "")  # dates handled by strategy

    print(f"\n{'='*70}")
    print("PHASE 2: Parameter Grid Search")
    print(f"{'='*70}")
    print(f"  Models: {len(models)}")
    print(f"  Configs: {len(param_grid)}")

    if returns_path.exists():
        print(f"\n  Returns cached: {returns_path}")
        returns_df = pd.read_parquet(returns_path)
    else:
        all_returns = []
        for i, model in enumerate(models):
            print(f"\n  [{i+1}/{len(models)}] {model.model_id}")
            model_returns = strategy.run_model_year(model, data, param_grid)
            if not model_returns.empty:
                all_returns.append(model_returns)
                print(f"    {len(model_returns)} strategy-days")
        if not all_returns:
            raise RuntimeError(
                "Phase 2: all models returned empty results. "
                "Check data availability (DVOL, spot bars, etc)."
            )
        returns_df = pd.concat(all_returns, ignore_index=True)
        returns_df.to_parquet(returns_path)
        print(f"\n  Saved: {returns_path} ({len(returns_df)} rows)")

    if features_path.exists():
        print(f"  Features cached: {features_path}")
        features_df = pd.read_parquet(features_path)
    else:
        print(f"\n  Computing daily features...")
        features_df = _compute_all_features(strategy, models, data, returns_df)
        features_df.to_parquet(features_path)
        print(f"  Saved: {features_path}")

    return returns_df, features_df


def _compute_all_features(strategy, models, data, returns_df):
    """Compute daily features for all models on all trading days."""
    trading_days = sorted(returns_df["date"].unique())
    daily_prices = strategy.get_daily_prices(data, models)
    feature_names = strategy.daily_feature_names()

    all_features = []
    for model in models:
        for day in trading_days:
            feat_vec = strategy.compute_features(model, day, data)
            if feat_vec is not None:
                row = {"model_id": model.model_id, "date": day}
                for fn, fv in zip(feature_names, feat_vec):
                    row[fn] = fv
                all_features.append(row)

    return pd.DataFrame(all_features)


def run_phase3(strategy: TradingStrategy,
               returns_df: pd.DataFrame,
               features_df: pd.DataFrame,
               output_dir: Path,
               model_group_fn=None,
               ) -> dict[str, dict]:
    """
    Phase 3: Train Random Forest for each model.

    model_group_fn: optional function mapping model_id → group key
        for pre-filtering. If provided, only model groups with positive
        average gross return in training data get RF models trained.
        This eliminates structurally losing model types BEFORE training.
    """
    models = strategy.get_models()
    param_grid = strategy.get_param_grid()
    feature_names = strategy.daily_feature_names()
    config_names = strategy.config_param_names()

    # Pre-filter: remove structurally unprofitable model types
    kept_ids = None
    if model_group_fn is not None:
        kept_ids, group_stats = filter_models_by_training(
            returns_df, group_fn=model_group_fn
        )
        print(f"\n  Model type pre-filter (training data profitability):")
        for gkey, gs in group_stats.items():
            status = "KEEP" if gs["kept"] else "DROP"
            print(f"    {gkey:>12s}: mean_ret={gs['mean_return']:+.6f}, "
                  f"n_models={gs['n_models']}, {status}")
        n_kept = len(kept_ids)
        n_total = len(models)
        print(f"  Kept {n_kept}/{n_total} models "
              f"({n_total - n_kept} filtered)")
    else:
        kept_ids = [m.model_id for m in models]

    print(f"\n{'='*70}")
    print("PHASE 3: Random Forest Training")
    n_configs = len(param_grid)
    n_days = features_df["date"].nunique() if not features_df.empty else 0
    print(f"  Samples per model: ~{n_configs} configs × ~{n_days} days")
    print(f"{'='*70}")

    kept_set = set(kept_ids)
    trained = {}
    n_trained = 0
    for i, model in enumerate(models):
        if model.model_id not in kept_set:
            trained[model.model_id] = {"model": None, "error": "filtered by pre-filter"}
            continue
        result = train_conditional_model(
            features_df, returns_df, model.model_id,
            param_grid, feature_names, config_names,
        )
        trained[model.model_id] = result
        status = (f"AUC={result['train_score']:.4f}, "
                  f"base_rate={result['base_rate']:.1%}, "
                  f"n={result['n_samples']}"
                  if result["model"] else result.get("error", "failed"))
        print(f"  [{i+1}/{len(models)}] {model.model_id}: {status}")
        if result.get("model"):
            n_trained += 1

    print(f"\n  Trained: {n_trained} models")

    # Save feature importances
    importance_summary = {mid: m.get("feature_importance", {})
                         for mid, m in trained.items() if m.get("feature_importance")}
    with open(output_dir / "phase3_importances.json", "w") as f:
        json.dump(importance_summary, f, indent=2)

    return trained


def run_phase4(strategy: TradingStrategy,
               trained_models: dict[str, dict],
               oos_data: dict,
               output_dir: Path,
               max_leverage: float = 2.0,
               max_weight_per_model: float = 0.05,
               prob_threshold: float = 0.50,
               min_lift: float = 0.0,
               allocation_mode: str = "equal_weight",
               corr_threshold: float = 0.85,
               warmup_daily: dict[str, pd.Series] | None = None,
               ) -> pd.DataFrame:
    """
    Phase 4: OOS portfolio trading.

    allocation_mode:
        "equal_weight" — for short-lived models (crypto TA). Default.
        "kelly" — for persistent models with return history (pairs, futures).

    Gate logic:
        If min_lift > 0: P > model's base_rate + min_lift (lift-based gate).
            Only trades when RF found a config meaningfully better than average.
        Else: P > prob_threshold (absolute gate, backward compatible).
    """
    models = strategy.get_models()
    param_grid = strategy.get_param_grid()

    gate_desc = (f"P > base_rate + {min_lift:.2f}" if min_lift > 0
                 else f"P > {prob_threshold:.2f}")
    print(f"\n{'='*70}")
    print("PHASE 4: OOS Portfolio Trading")
    print(f"  Mode: {allocation_mode}, Gate: {gate_desc}")
    print(f"  Max leverage: {max_leverage}, Max weight/model: {max_weight_per_model}")
    print(f"{'='*70}")

    daily_prices = strategy.get_daily_prices(oos_data, models)
    if warmup_daily:
        daily_prices = strategy.prepare_warmup(daily_prices, warmup_daily)
        print(f"  Warmup applied")

    trading_days = strategy.get_trading_days(oos_data)
    print(f"  Trading days: {len(trading_days)}")

    returns_history = []  # accumulated for kelly mode
    portfolio_pnl = []
    model_returns = {m.model_id: [] for m in models}
    model_predictions = {m.model_id: [] for m in models}

    for day_idx, day in enumerate(trading_days):
        day_str = day.strftime("%Y-%m-%d") if hasattr(day, 'strftime') else str(day)

        # Step 1: RF prediction for each model
        predictions = []
        n_no_model = n_no_feat = 0
        for model in models:
            mid = model.model_id
            tm = trained_models.get(mid, {})
            if tm.get("model") is None:
                n_no_model += 1
                continue

            feat_vec = strategy.compute_features(model, day_str, oos_data)
            if feat_vec is None:
                n_no_feat += 1
                continue

            config, p_prof, exp_ret = predict_model(tm, feat_vec, param_grid)
            base_rate = tm.get("base_rate", 0.5)
            predictions.append({
                "model_id": mid,
                "model": model,
                "config": config,
                "p_profitable": p_prof,
                "expected_return": exp_ret,
                "base_rate": base_rate,
            })

        # Diagnostics
        if day_idx < 5 or day_idx % 10 == 0:
            n_above = 0
            if predictions:
                probs = [p["p_profitable"] for p in predictions]
                if min_lift > 0:
                    n_above = sum(1 for p in predictions
                                  if p["p_profitable"] > p["base_rate"] + min_lift)
                else:
                    n_above = sum(1 for p in probs if p > prob_threshold)
            print(f"  Day {day_idx+1} ({day_str}): "
                  f"predicted={len(predictions)}, "
                  f"above_gate={n_above}")

        if not predictions:
            continue

        # Step 2: Allocate
        hist_df = pd.DataFrame(returns_history) if returns_history else pd.DataFrame()
        allocation = compute_allocation(
            predictions, hist_df,
            max_leverage=max_leverage,
            max_weight_per_model=max_weight_per_model,
            prob_threshold=prob_threshold,
            min_lift=min_lift,
            mode=allocation_mode,
            corr_threshold=corr_threshold,
        )

        if day_idx < 5 or (day_idx < 35 and day_idx % 5 == 0):
            n_alloc = len(allocation)
            alloc_sum = sum(allocation.values()) if allocation else 0
            print(f"    Allocated: {n_alloc} models, "
                  f"total_weight={alloc_sum:.3f}, "
                  f"per_model={alloc_sum/max(n_alloc,1):.4f}")

        # Step 3: Execute
        day_pnl = 0.0
        n_executed = 0
        for pred in predictions:
            mid = pred["model_id"]
            weight = allocation.get(mid, 0)
            if weight < 0.0001:
                continue

            result = strategy.run_single_day(
                pred["model"], pred["config"], day, oos_data
            )
            day_pnl += result["daily_return"] * weight

            # Accumulate history (used by kelly mode for covariance)
            returns_history.append({
                "date": day_str, "model_id": mid,
                "daily_return": result["daily_return"],
            })

            model_returns[mid].append(result["daily_return"])
            actual = 1.0 if result["daily_return"] > 0 else 0.0
            model_predictions[mid].append((pred["p_profitable"], actual))
            n_executed += 1

        if day_idx < 5 or (day_idx < 35 and day_idx % 5 == 0):
            print(f"    Executed: {n_executed}")

        portfolio_pnl.append({
            "date": day_str,
            "portfolio_return": day_pnl,
            "n_models_active": n_executed,
            "total_weight": sum(allocation.get(p["model_id"], 0) for p in predictions),
        })

        if (day_idx + 1) % 20 == 0:
            cum_ret = sum(p["portfolio_return"] for p in portfolio_pnl)
            print(f"  Day {day_idx+1}/{len(trading_days)}: "
                  f"cum={cum_ret:+.4f}, active={n_executed}")

    # ── Reporting ─────────────────────────────────────────────────
    pnl_df = pd.DataFrame(portfolio_pnl)
    _print_phase4_report(pnl_df, model_returns, model_predictions, models, output_dir)

    pnl_df.to_parquet(output_dir / "phase4_portfolio.parquet")
    return pnl_df


def _print_phase4_report(pnl_df, model_returns, model_predictions,
                         models, output_dir):
    """Print comprehensive Phase 4 OOS report."""
    if pnl_df.empty:
        print("  No trading days.")
        return

    rets = pnl_df["portfolio_return"].values
    n_days = len(rets)
    daily_mean = np.mean(rets)
    daily_std = np.std(rets) + 1e-10
    sr = daily_mean / daily_std * np.sqrt(252)
    ann_ret = daily_mean * 252
    ann_vol = daily_std * np.sqrt(252)
    cum = np.cumsum(rets)
    max_dd = np.min(cum - np.maximum.accumulate(cum))
    win_days = (rets > 0).sum()

    print(f"\n  OOS Results:")
    print(f"    Trading days:     {n_days}")
    print(f"    Cumulative ret:   {cum[-1]:+.4f}  ({cum[-1]*100:+.2f}%)")
    print(f"    Ann. return:      {ann_ret:+.4f}  ({ann_ret*100:+.1f}%)")
    print(f"    Ann. volatility:  {ann_vol:.4f}   ({ann_vol*100:.1f}%)")
    print(f"    Sharpe ratio:     {sr:+.4f}")
    print(f"    Max drawdown:     {max_dd:+.4f}  ({max_dd*100:+.2f}%)")
    print(f"    Win days:         {win_days}/{n_days} ({win_days/n_days:.1%})")
    print(f"    Avg models/day:   {pnl_df['n_models_active'].mean():.1f}")

    # Calibration
    all_pred, all_actual = [], []
    for mid in model_predictions:
        for (pred_p, actual_win) in model_predictions[mid]:
            if np.isfinite(pred_p) and np.isfinite(actual_win):
                all_pred.append(pred_p)
                all_actual.append(actual_win)

    if len(all_pred) > 10:
        all_pred = np.array(all_pred)
        all_actual = np.array(all_actual)
        actual_rate = np.mean(all_actual)
        accuracy = np.mean((all_pred > 0.5).astype(int) == all_actual)

        print(f"\n  OOS Prediction Calibration:")
        print(f"    Samples: {len(all_pred)}, Base win rate: {actual_rate:.1%}")
        print(f"    Accuracy: {accuracy:.1%}, Brier: {np.mean((all_pred - all_actual)**2):.4f}")
        print(f"\n    {'P bin':>15s}  {'n':>5s}  {'Actual WR':>9s}  {'Lift':>6s}")
        for lo, hi in [(0.0, 0.4), (0.4, 0.5), (0.5, 0.55), (0.55, 0.6),
                       (0.6, 0.65), (0.65, 0.7), (0.7, 0.8), (0.8, 1.01)]:
            mask = (all_pred >= lo) & (all_pred < hi)
            if mask.sum() >= 3:
                bin_wr = np.mean(all_actual[mask])
                print(f"    [{lo:.2f}, {hi:.2f})  {mask.sum():5d}  "
                      f"{bin_wr:.1%}  {bin_wr - actual_rate:+.1%}")

    # Per-model breakdown
    print(f"\n  Per-Model Performance (top 20 by Sharpe):")
    print(f"    {'Model':>20s}  {'Days':>5s}  {'MeanRet':>9s}  {'Sharpe':>7s}  "
          f"{'CumRet':>8s}  {'WinRate':>7s}  {'AvgP':>6s}")

    model_stats = []
    for model in models:
        mid = model.model_id
        rets_m = np.array(model_returns.get(mid, []))
        preds_m = model_predictions.get(mid, [])
        if len(rets_m) < 3:
            continue
        m_sr = np.mean(rets_m) / (np.std(rets_m) + 1e-10) * np.sqrt(252)
        avg_p = np.mean([p[0] for p in preds_m]) if preds_m else 0.5
        model_stats.append({
            "model_id": mid, "n_days": len(rets_m),
            "mean_ret": np.mean(rets_m), "sharpe": m_sr,
            "cum_ret": np.sum(rets_m), "win_rate": np.mean(rets_m > 0),
            "avg_p": avg_p,
        })

    model_stats.sort(key=lambda x: x["sharpe"], reverse=True)
    for ms in model_stats[:20]:
        print(f"    {ms['model_id']:>20s}  {ms['n_days']:5d}  "
              f"{ms['mean_ret']:+.6f}  {ms['sharpe']:+.3f}  "
              f"{ms['cum_ret']:+.5f}  {ms['win_rate']:.1%}  "
              f"{ms['avg_p']:.3f}")

    if len(model_stats) > 5:
        print(f"\n  Bottom 5:")
        for ms in model_stats[-5:]:
            print(f"    {ms['model_id']:>20s}  {ms['n_days']:5d}  "
                  f"{ms['mean_ret']:+.6f}  {ms['sharpe']:+.3f}  "
                  f"{ms['cum_ret']:+.5f}  {ms['win_rate']:.1%}  "
                  f"{ms['avg_p']:.3f}")

    if model_stats:
        sharpes = [ms["sharpe"] for ms in model_stats]
        print(f"\n  Model Sharpe Distribution (n={len(model_stats)}):")
        print(f"    Mean: {np.mean(sharpes):+.3f}  Median: {np.median(sharpes):+.3f}")
        print(f"    Positive: {sum(1 for s in sharpes if s > 0)}/{len(sharpes)}")

    with open(output_dir / "phase4_model_stats.json", "w") as f:
        json.dump(model_stats, f, indent=2)
