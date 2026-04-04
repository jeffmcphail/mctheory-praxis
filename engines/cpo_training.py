"""
CPO Training — Minute-Frequency Features + Regime Matrix Integration.

Drop-in replacement for the feature computation and training pipeline
in pairs_trading.py. Uses:
    - engines/minute_features.py (112 features per the Chan paper spec)
    - engines/regime_engine.py (12 regime classes + ~40 raw sub-features)

The Phase 2 returns computation (grid search on minute data) is unchanged.
Only the feature computation, training, and prediction are rebuilt.

Feature modes:
    "chan"    — 112 minute-bar indicators (paper spec)
    "regime"  — 12 regime state integers + ~40 raw sub-features
    "hybrid"  — chan + regime combined (~160 features)
    "ablation" — single regime class (for ablation experiments)

Usage:
    from engines.cpo_training import compute_features_v2, train_v2, predict_v2

    # Phase 2b: Compute features
    features_df = compute_features_v2(
        pairs, minute_data, trading_days,
        mode="chan",  # or "regime", "hybrid", "ablation:D"
    )

    # Phase 3: Train RF (same interface, wider features)
    model = train_v2(features_df, returns_df, pair_id, param_grid)

    # Phase 4: Predict (same interface)
    config, p, e_ret = predict_v2(model, features_row, param_grid)
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from engines.pairs_trading import (
    PairSpec, ParamConfig,
    generate_param_grid,
    CONFIG_PARAM_NAMES,
    _config_to_features,
)
from engines.minute_features import (
    compute_minute_features,
    FEATURE_LOOKBACKS,
    INDICATOR_NAMES,
    feature_column_names,
)
from engines.regime_engine import RegimeEngine

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE COMPUTATION — V2
# ═════════════════════════════════════════════════════════════════════════════

VALID_MODES = {"chan", "regime", "hybrid"}


def _parse_mode(mode: str) -> tuple[str, str | None]:
    """
    Parse feature mode string.

    Returns (base_mode, ablation_class).
    Examples:
        "chan" → ("chan", None)
        "regime" → ("regime", None)
        "ablation:D" → ("ablation", "D")
        "ablation:BCD" → ("ablation", "BCD")
    """
    if mode.startswith("ablation:"):
        classes = mode.split(":", 1)[1].upper()
        return "ablation", classes
    return mode.lower(), None


def compute_features_v2(
    pairs: list[PairSpec],
    minute_data: dict[str, pd.DataFrame],
    trading_days: list[str],
    mode: str = "chan",
    universe_ohlcv: dict[str, pd.DataFrame] | None = None,
    funding_data: dict[str, np.ndarray] | None = None,
    bars_per_day: int = 24,
    feature_lookbacks: list[int] | None = None,
    market_close_time: str = "16:00",
) -> pd.DataFrame:
    """
    Compute features for all pairs × all trading days.

    This replaces the old compute_daily_features loop in run_phase2_training.

    Args:
        pairs: List of PairSpec from Phase 1.
        minute_data: Dict of ticker → minute OHLCV DataFrame.
        trading_days: List of date strings (YYYY-MM-DD).
        mode: Feature mode — "chan", "regime", "hybrid", or "ablation:XYZ".
        universe_ohlcv: Multi-asset hourly data for regime classes H, K.
        funding_data: Dict of asset → funding rate array for regime classes F, J.
        bars_per_day: 24 for crypto, 7 for equities (used by regime engine).
        feature_lookbacks: Override Chan paper lookback windows.
        market_close_time: When to evaluate features (default 16:00).

    Returns:
        DataFrame with columns: pair_id, date, and feature columns.
        Feature columns depend on mode:
            "chan": 112 minute-frequency indicator features
            "regime": ~52 regime features (12 states + ~40 raw)
            "hybrid": ~164 combined features
            "ablation:D": features from regime class D only
    """
    base_mode, ablation_classes = _parse_mode(mode)

    if feature_lookbacks is None:
        feature_lookbacks = FEATURE_LOOKBACKS

    use_chan = base_mode in ("chan", "hybrid")
    use_regime = base_mode in ("regime", "hybrid", "ablation")

    regime_engine = None
    if use_regime:
        regime_engine = RegimeEngine(bars_per_day=bars_per_day)

    all_rows = []
    n_total = len(pairs) * len(trading_days)
    n_done = 0
    t0 = time.time()

    for pair in pairs:
        target_bars = minute_data.get(pair.target)
        hedge_bars = minute_data.get(pair.hedge)

        if target_bars is None or hedge_bars is None:
            logger.warning("Missing minute data for %s", pair.pair_id)
            continue

        for day_str in trading_days:
            n_done += 1

            row = {"pair_id": pair.pair_id, "date": day_str}

            # Compute as-of timestamp (previous day's close — no lookahead)
            as_of = _prev_close_timestamp(day_str, market_close_time,
                                          target_bars.index)
            if as_of is None:
                continue

            # ── Chan minute features (112) ──────────────────────────
            if use_chan:
                chan_feats = compute_minute_features(
                    target_bars, hedge_bars,
                    as_of_timestamp=as_of,
                    lookback_windows=feature_lookbacks,
                )
                if chan_feats is None:
                    continue
                row.update(chan_feats)

            # ── Regime features ─────────────────────────────────────
            if use_regime and regime_engine is not None:
                # Use target asset's hourly data for regime computation
                # Aggregate minute → hourly if needed
                hourly = _minute_to_hourly(target_bars, as_of)
                if hourly is not None and len(hourly) >= bars_per_day * 35:
                    # Get funding data for this asset if available
                    fr = funding_data.get(pair.target) if funding_data else None

                    regime_state = regime_engine.compute(
                        ohlcv_hourly=hourly,
                        funding_rates=fr,
                        universe_ohlcv=universe_ohlcv,
                    )

                    regime_row = regime_state.to_feature_row()

                    # Filter to ablation classes if specified
                    if base_mode == "ablation" and ablation_classes:
                        filtered = {}
                        for k, v in regime_row.items():
                            # Keep regime state columns for specified classes
                            if k.startswith("regime_"):
                                class_name = k.replace("regime_", "")
                                # Map class name back to letter
                                from engines.regime_engine import REGIME_CLASS_NAMES
                                letter = None
                                for l, name in REGIME_CLASS_NAMES.items():
                                    if name == class_name:
                                        letter = l
                                        break
                                if letter and letter in ablation_classes:
                                    filtered[k] = v
                            # Keep raw features for specified classes
                            elif "_" in k:
                                prefix = k.split("_")[0]
                                if prefix in ablation_classes:
                                    filtered[k] = v
                        regime_row = filtered

                    row.update(regime_row)
                elif not use_chan:
                    # Regime-only mode but insufficient data — skip
                    continue

            all_rows.append(row)

        if n_done % 500 == 0:
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            logger.info("Features: %d/%d (%.0f/sec)", n_done, n_total, rate)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Drop any all-NaN feature columns
    feat_cols = [c for c in df.columns if c not in ("pair_id", "date")]
    df[feat_cols] = df[feat_cols].fillna(0.0)

    logger.info("Feature matrix: %d rows × %d features (mode=%s)",
                len(df), len(feat_cols), mode)

    return df


def _prev_close_timestamp(
    day_str: str,
    market_close_time: str,
    index: pd.DatetimeIndex,
) -> pd.Timestamp | None:
    """Find the previous trading day's close timestamp."""
    day = pd.Timestamp(day_str)

    for offset in range(1, 6):
        check = day - pd.Timedelta(days=offset)
        ts = pd.Timestamp(f"{check.date()} {market_close_time}")
        if index.tz is not None:
            ts = ts.tz_localize(index.tz)
        mask = index <= ts
        if mask.sum() > 100:
            return ts

    return None


def _minute_to_hourly(
    minute_df: pd.DataFrame,
    as_of: pd.Timestamp,
) -> pd.DataFrame | None:
    """Resample minute bars to hourly up to as_of."""
    subset = minute_df[minute_df.index <= as_of]
    if len(subset) < 60:
        return None

    hourly = subset.resample("1h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["close"])

    return hourly


# ═════════════════════════════════════════════════════════════════════════════
# TRAINING — V2
# ═════════════════════════════════════════════════════════════════════════════

def get_feature_columns(features_df: pd.DataFrame) -> list[str]:
    """Extract feature column names (everything except pair_id and date)."""
    return [c for c in features_df.columns if c not in ("pair_id", "date")]


def train_v2(
    features_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    pair_id: str,
    param_grid: list[ParamConfig] | None = None,
    n_estimators: int = 200,
    max_depth: int = 5,
) -> dict:
    """
    Train RF classifier for one pair using v2 features.

    Same interface as pairs_trading train_conditional_model but works with
    any feature set (chan 112, regime ~52, hybrid ~164, ablation variable).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    if param_grid is None:
        param_grid = generate_param_grid()

    pair_features = features_df[features_df["pair_id"] == pair_id].copy()
    pair_returns = returns_df[returns_df["pair_id"] == pair_id].copy()

    if pair_features.empty or pair_returns.empty:
        return {"model": None, "error": "no data"}

    feature_cols = get_feature_columns(features_df)
    config_lookup = {c.config_id: _config_to_features(c) for c in param_grid}

    pair_features = pair_features.set_index("date")
    trading_dates = sorted(pair_returns["date"].unique())
    feature_dates = set(pair_features.index)

    X_rows = []
    y_rows = []

    for i in range(len(trading_dates) - 1):
        feat_date = trading_dates[i]
        trade_date = trading_dates[i + 1]

        if feat_date not in feature_dates:
            continue

        daily_feats = pair_features.loc[feat_date, feature_cols]
        if hasattr(daily_feats, "values"):
            daily_vec = daily_feats.values.astype(float)
        else:
            daily_vec = np.array([float(daily_feats)])

        if not np.all(np.isfinite(daily_vec)):
            continue

        day_returns = pair_returns[pair_returns["date"] == trade_date]
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

    valid = np.all(np.isfinite(X), axis=1)
    X, y = X[valid], y[valid]

    if len(X) < 100:
        return {"model": None, "error": f"only {len(X)} valid samples (need 100+)"}

    n_pos = int(y.sum())
    base_rate = n_pos / len(y)

    all_feature_names = feature_cols + [f"cfg_{n}" for n in CONFIG_PARAM_NAMES]

    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=20, min_samples_split=40,
        class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    rf.fit(X, y)

    proba_train = rf.predict_proba(X)[:, 1]
    try:
        auc = roc_auc_score(y, proba_train)
    except ValueError:
        auc = 0.5

    # Mean return by outcome
    ret_vals = np.array([r["daily_return"] for _, r in pair_returns.iterrows()
                         if np.isfinite(r["daily_return"])])
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
        "n_features": len(feature_cols),
    }


def predict_v2(
    trained_model: dict,
    features: np.ndarray,
    param_grid: list[ParamConfig],
) -> tuple[ParamConfig, float, float]:
    """
    Predict P(profitable) for all configs, pick the best.

    Same interface as pairs_trading predict_model but works with v2 features.
    """
    model = trained_model.get("model")
    if model is None:
        return param_grid[0], 0.0, 0.0

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    X_pred = []
    for config in param_grid:
        config_vec = _config_to_features(config)
        X_pred.append(np.concatenate([features, config_vec]))

    X_pred = np.array(X_pred)
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
# FULL PIPELINE — V2
# ═════════════════════════════════════════════════════════════════════════════

def run_phase2b_features(
    pairs: list[PairSpec],
    minute_data: dict[str, pd.DataFrame],
    returns_df: pd.DataFrame,
    output_dir: Path,
    mode: str = "chan",
    **kwargs,
) -> pd.DataFrame:
    """
    Phase 2b: Compute v2 features for all pairs × all training days.

    Replaces the feature computation part of run_phase2_training.
    The returns_df (from Phase 2) is unchanged.
    """
    cache_path = output_dir / f"phase2b_features_{mode.replace(':', '_')}.parquet"

    if cache_path.exists():
        print(f"  Features cached: {cache_path}")
        return pd.read_parquet(cache_path)

    trading_days = sorted(returns_df["date"].unique())
    print(f"\n  Computing v2 features (mode={mode})...")
    print(f"  Pairs: {len(pairs)}, Days: {len(trading_days)}")

    t0 = time.time()
    features_df = compute_features_v2(
        pairs, minute_data, trading_days, mode=mode, **kwargs
    )
    elapsed = time.time() - t0

    print(f"  Features: {features_df.shape[0]} rows × "
          f"{len(get_feature_columns(features_df))} features ({elapsed:.1f}s)")

    output_dir.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(cache_path)
    print(f"  Saved: {cache_path}")

    return features_df


def run_phase3_v2(
    pairs: list[PairSpec],
    returns_df: pd.DataFrame,
    features_df: pd.DataFrame,
    output_dir: Path,
    param_grid: list[ParamConfig] | None = None,
    mode: str = "chan",
) -> dict[str, dict]:
    """
    Phase 3 v2: Train RF for each pair with v2 features.
    """
    if param_grid is None:
        param_grid = generate_param_grid()

    feature_cols = get_feature_columns(features_df)
    print(f"\n{'='*70}")
    print(f"PHASE 3 v2: RF Training (mode={mode}, {len(feature_cols)} features)")
    print(f"{'='*70}")

    models = {}
    for i, pair in enumerate(pairs):
        result = train_v2(features_df, returns_df, pair.pair_id, param_grid)
        models[pair.pair_id] = result
        if result["model"]:
            print(f"  [{i+1}/{len(pairs)}] {pair.pair_id}: "
                  f"AUC={result['train_score']:.4f}, "
                  f"base_rate={result['base_rate']:.1%}, "
                  f"n={result['n_samples']}, "
                  f"features={result['n_features']}")
        else:
            print(f"  [{i+1}/{len(pairs)}] {pair.pair_id}: {result.get('error')}")

    # Save importances
    importance_summary = {pid: m["feature_importance"]
                          for pid, m in models.items()
                          if m.get("feature_importance")}

    suffix = mode.replace(":", "_")
    imp_path = output_dir / f"phase3_importances_{suffix}.json"
    with open(imp_path, "w") as f:
        json.dump(importance_summary, f, indent=2)

    return models


def run_ablation_experiment(
    pairs: list[PairSpec],
    minute_data: dict[str, pd.DataFrame],
    returns_df: pd.DataFrame,
    output_dir: Path,
    param_grid: list[ParamConfig] | None = None,
    regime_classes: str = "ABCDEFGHIJKL",
    bars_per_day: int = 24,
    **kwargs,
) -> pd.DataFrame:
    """
    Run regime ablation: train CPO with each regime class independently.

    For each class in regime_classes:
        1. Compute features using only that class
        2. Train RF
        3. Record AUC and calibration metrics

    Returns a summary DataFrame with one row per (class, pair).
    """
    if param_grid is None:
        param_grid = generate_param_grid()

    from engines.regime_engine import REGIME_CLASS_NAMES

    print(f"\n{'='*70}")
    print("REGIME ABLATION EXPERIMENT")
    print(f"  Classes to test: {regime_classes}")
    print(f"  Pairs: {len(pairs)}")
    print(f"{'='*70}")

    results = []

    # Baseline: config params only (no market features)
    print(f"\n── Baseline (config params only) ──")
    baseline_features = _make_config_only_features(returns_df)
    baseline_models = {}
    for pair in pairs:
        model = train_v2(baseline_features, returns_df, pair.pair_id, param_grid)
        baseline_models[pair.pair_id] = model
        if model["model"]:
            results.append({
                "class": "baseline",
                "class_name": "config_only",
                "pair_id": pair.pair_id,
                "auc": model["train_score"],
                "base_rate": model["base_rate"],
                "n_samples": model["n_samples"],
                "n_features": 0,  # no market features
            })
            print(f"  {pair.pair_id}: AUC={model['train_score']:.4f}")

    # Each regime class independently
    for cls in regime_classes:
        cls_name = REGIME_CLASS_NAMES.get(cls, cls)
        mode = f"ablation:{cls}"
        print(f"\n── Class {cls}: {cls_name} ──")

        try:
            features_df = compute_features_v2(
                pairs, minute_data,
                sorted(returns_df["date"].unique()),
                mode=mode,
                bars_per_day=bars_per_day,
                **kwargs,
            )

            feat_cols = get_feature_columns(features_df)
            if not feat_cols:
                print(f"  No features computed — skipping")
                continue

            for pair in pairs:
                model = train_v2(features_df, returns_df, pair.pair_id, param_grid)
                if model["model"]:
                    results.append({
                        "class": cls,
                        "class_name": cls_name,
                        "pair_id": pair.pair_id,
                        "auc": model["train_score"],
                        "base_rate": model["base_rate"],
                        "n_samples": model["n_samples"],
                        "n_features": model["n_features"],
                    })
                    # AUC lift over baseline
                    baseline_auc = baseline_models.get(pair.pair_id, {}).get("train_score", 0.5)
                    lift = model["train_score"] - baseline_auc
                    print(f"  {pair.pair_id}: AUC={model['train_score']:.4f} "
                          f"(lift={lift:+.4f}, features={model['n_features']})")

        except Exception as e:
            print(f"  Failed: {e}")
            continue

    # Full regime (all classes)
    print(f"\n── Full regime (all classes) ──")
    try:
        full_features = compute_features_v2(
            pairs, minute_data,
            sorted(returns_df["date"].unique()),
            mode="regime",
            bars_per_day=bars_per_day,
            **kwargs,
        )
        for pair in pairs:
            model = train_v2(full_features, returns_df, pair.pair_id, param_grid)
            if model["model"]:
                results.append({
                    "class": "full",
                    "class_name": "all_regime",
                    "pair_id": pair.pair_id,
                    "auc": model["train_score"],
                    "base_rate": model["base_rate"],
                    "n_samples": model["n_samples"],
                    "n_features": model["n_features"],
                })
                baseline_auc = baseline_models.get(pair.pair_id, {}).get("train_score", 0.5)
                lift = model["train_score"] - baseline_auc
                print(f"  {pair.pair_id}: AUC={model['train_score']:.4f} "
                      f"(lift={lift:+.4f}, features={model['n_features']})")
    except Exception as e:
        print(f"  Failed: {e}")

    summary = pd.DataFrame(results)
    summary_path = output_dir / "ablation_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\n  Saved: {summary_path}")

    # Print summary
    if not summary.empty:
        print(f"\n{'='*70}")
        print("ABLATION SUMMARY — Mean AUC by regime class")
        print(f"{'='*70}")
        mean_auc = summary.groupby(["class", "class_name"])["auc"].mean()
        mean_auc = mean_auc.sort_values(ascending=False)
        for (cls, name), auc in mean_auc.items():
            baseline = mean_auc.get(("baseline", "config_only"), 0.5)
            lift = auc - baseline
            print(f"  {cls:>8} ({name:>20}): AUC={auc:.4f} (lift={lift:+.4f})")

    return summary


def _make_config_only_features(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Create a dummy feature DataFrame with just pair_id and date (no market features)."""
    # We need at least one feature column for the training loop to work
    rows = []
    for pair_id in returns_df["pair_id"].unique():
        pair_data = returns_df[returns_df["pair_id"] == pair_id]
        for day in pair_data["date"].unique():
            rows.append({
                "pair_id": pair_id,
                "date": day,
                "dummy_const": 1.0,  # constant feature — RF baseline
            })
    return pd.DataFrame(rows)
