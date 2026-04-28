"""
engines/intrabar_predictor.py -- Intrabar multi-horizon confluence system

Operates on N-minute OHLCV bars (default 5) aggregated from 1-min data.
Predicts return distributions at 1/3/5/10/15 bar horizons (by default
5/15/25/50/75 minutes forward). Generates mean-reversion confluence
signals by combining z-score + dual-horizon LSTM alignment + Hurst regime.

This is a fork of engines/lstm_predictor.py adapted for intrabar timescales.
The daily predictor runs in parallel -- do NOT modify it.

Usage:
    python -m engines.intrabar_predictor build-features --asset BTC
    python -m engines.intrabar_predictor train --asset BTC
    python -m engines.intrabar_predictor predict --asset BTC
    python -m engines.intrabar_predictor confluence --asset BTC
    python -m engines.intrabar_predictor confluence --asset BTC --zscore 1.5
    python -m engines.intrabar_predictor backtest --asset BTC --zscore 2.0
"""
import argparse
import json
import math
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ===================================================================
# CONFIG
# ===================================================================
REPO_ROOT = Path(__file__).parent.parent
CRYPTO_DB_PATH = REPO_ROOT / "data" / "crypto_data.db"
MODEL_DIR = REPO_ROOT / "models" / "intrabar"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BAR_SIZE_MINUTES = 5          # Aggregate 1-min data to this size before training
SEQUENCE_LENGTH = 60          # 60 bars of lookback (at 5-min bars = 5 hours)
# At 5-min bars, horizons translate to: 5, 15, 25, 50, 75 minutes forward
MULTI_HORIZONS = [1, 3, 5, 10, 15]
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
TRAIN_SPLIT = 0.8

SUPPORTED_ASSETS = {"BTC", "ETH"}

# Minimum bars needed before we can compute all features
MIN_BARS_FOR_FEATURES = 120  # Need 120 bars for longest Hurst window

# Fee presets for backtest cost sensitivity (per-side, fractional units)
FEE_PRESETS = {
    "maker":   0.0005,   # 0.05% per side -- Binance VIP maker (conservative)
    "taker":   0.0010,   # 0.10% per side -- Binance standard taker
    "default": 0.0015,   # 0.15% per side -- v2 assumption (0.10% exchange + 0.05% slip)
    "zero":    0.0,      # rebate venue / paper trading
}

# Hurst regime thresholds (heuristic)
HURST_REVERTING = 0.45
HURST_TRENDING = 0.55


# ===================================================================
# DATA LOADING
# ===================================================================

def load_intrabar_data(asset, bar_minutes=BAR_SIZE_MINUTES, limit_bars=None):
    """Load 1-min OHLCV and aggregate to `bar_minutes` bars.

    Aggregation rules:
      - open  = first open of the group
      - high  = max high of the group
      - low   = min low of the group
      - close = last close of the group
      - volume = sum of volumes
      - timestamp/datetime = first timestamp of the group (bar start)

    Drops any group with fewer than `bar_minutes` rows (incomplete bars --
    this naturally handles exchange-downtime gaps and zero-volume minutes).
    When bar_minutes=1, returns the 1-min data directly.
    """
    if not CRYPTO_DB_PATH.exists():
        print(f"  No database at {CRYPTO_DB_PATH}")
        print(f"  Run: python -m engines.crypto_data_collector collect-all --asset {asset}")
        return []

    conn = sqlite3.connect(str(CRYPTO_DB_PATH))
    conn.row_factory = sqlite3.Row
    query = """
        SELECT timestamp, datetime, open, high, low, close, volume
        FROM ohlcv_1m
        WHERE asset = ?
        ORDER BY timestamp ASC
    """
    rows = conn.execute(query, (asset,)).fetchall()
    conn.close()

    raw = [dict(r) for r in rows]
    filtered = [r for r in raw if r["volume"] > 0]
    dropped_zero_vol = len(raw) - len(filtered)

    if bar_minutes == 1:
        clean = filtered
    else:
        bar_seconds = bar_minutes * 60
        aggregated = []
        current_group = []
        current_bar_start = None

        for r in filtered:
            ts = r["timestamp"]
            bar_start = (ts // bar_seconds) * bar_seconds

            if current_bar_start is None:
                current_bar_start = bar_start
                current_group = [r]
            elif bar_start == current_bar_start:
                current_group.append(r)
            else:
                if len(current_group) == bar_minutes:
                    aggregated.append(_aggregate_bars(current_group))
                current_bar_start = bar_start
                current_group = [r]

        if len(current_group) == bar_minutes:
            aggregated.append(_aggregate_bars(current_group))

        clean = aggregated

    if limit_bars and len(clean) > limit_bars:
        clean = clean[-limit_bars:]

    if clean:
        if bar_minutes == 1:
            print(f"  Loaded {len(clean)} 1-min bars for {asset} "
                  f"(dropped {dropped_zero_vol} zero-vol)")
        else:
            print(f"  Aggregated {len(filtered)} 1-min bars into "
                  f"{len(clean)} {bar_minutes}-min bars for {asset} "
                  f"(dropped {dropped_zero_vol} zero-vol)")
        print(f"  Range: {clean[0]['datetime']} to {clean[-1]['datetime']}")
    else:
        print(f"  No data for {asset}")

    return clean


def _aggregate_bars(group):
    """Aggregate a list of 1-min bar rows into a single larger bar."""
    return {
        "timestamp": group[0]["timestamp"],
        "datetime": group[0]["datetime"],
        "open": group[0]["open"],
        "high": max(r["high"] for r in group),
        "low": min(r["low"] for r in group),
        "close": group[-1]["close"],
        "volume": sum(r["volume"] for r in group),
    }


# ===================================================================
# HURST ESTIMATION (from lstm_predictor.py)
# ===================================================================

def estimate_hurst(prices, min_window=10):
    """Rescaled Range (R/S) analysis for Hurst exponent.

    H > 0.5: Trending (momentum regime)
    H = 0.5: Random walk (no edge)
    H < 0.5: Anti-persistent, mean-reverting
    """
    n = len(prices)
    if n < min_window * 2:
        return 0.5

    returns = np.diff(np.log(np.array(prices, dtype=np.float64)))
    returns = returns[np.isfinite(returns)]

    if len(returns) < min_window * 2:
        return 0.5

    window_sizes = []
    w = min_window
    while w <= len(returns) // 2:
        window_sizes.append(w)
        w = int(w * 1.5)

    if len(window_sizes) < 3:
        return 0.5

    rs_values = []
    for w in window_sizes:
        n_windows = len(returns) // w
        if n_windows == 0:
            continue

        rs_list = []
        for i in range(n_windows):
            segment = returns[i * w:(i + 1) * w]
            mean = segment.mean()
            deviations = np.cumsum(segment - mean)
            r = deviations.max() - deviations.min()
            s = segment.std(ddof=1)
            if s > 0:
                rs_list.append(r / s)

        if rs_list:
            rs_values.append((w, np.mean(rs_list)))

    if len(rs_values) < 3:
        return 0.5

    log_n = np.log([v[0] for v in rs_values])
    log_rs = np.log([v[1] for v in rs_values])

    n_pts = len(log_n)
    sum_x = log_n.sum()
    sum_y = log_rs.sum()
    sum_xy = (log_n * log_rs).sum()
    sum_x2 = (log_n ** 2).sum()

    denom = n_pts * sum_x2 - sum_x ** 2
    if abs(denom) < 1e-12:
        return 0.5

    H = (n_pts * sum_xy - sum_x * sum_y) / denom
    return float(np.clip(H, 0.01, 0.99))


# ===================================================================
# FEATURE ENGINEERING
# ===================================================================

def compute_rolling_zscore(closes, window=60):
    """Rolling z-score -- the primary mean-reversion signal."""
    arr = np.array(closes)
    if len(arr) < window:
        return 0.0
    window_data = arr[-window:]
    mean = window_data.mean()
    std = window_data.std()
    if std == 0:
        return 0.0
    return float((arr[-1] - mean) / std)


def compute_intrabar_features(records, idx):
    """Compute features at bar index idx (backward-looking only).

    Features:
    - Multi-lookback returns (1, 5, 15, 30, 60 bars)
    - Price vs moving averages (5, 15, 30, 60 bars)
    - Realized volatility (15, 30, 60 bars)
    - RSI-14 (on 1-min bars)
    - Bollinger band position (20-bar)
    - Volume ratio (current vs 30-bar mean)
    - Multi-timescale Hurst (30, 60, 120 bar windows)
    - Rolling z-score (20, 60, 120 bar windows)
    """
    if idx < MIN_BARS_FOR_FEATURES:
        return None

    rec = records[idx]
    close = rec["close"]

    # Price and volume history (up to 200 bars back)
    lookback_start = max(0, idx - 200)
    closes = [records[i]["close"] for i in range(lookback_start, idx + 1)]
    volumes = [records[i]["volume"] for i in range(lookback_start, idx + 1)]
    highs = [records[i]["high"] for i in range(lookback_start, idx + 1)]
    lows = [records[i]["low"] for i in range(lookback_start, idx + 1)]

    features = {}

    # -- Multi-lookback returns --
    for w in [1, 5, 15, 30, 60]:
        if len(closes) > w:
            features[f"return_{w}bar"] = (closes[-1] - closes[-1 - w]) / closes[-1 - w] * 100

    # -- Price vs moving averages --
    for w in [5, 15, 30, 60]:
        if len(closes) >= w:
            ma = sum(closes[-w:]) / w
            features[f"price_vs_ma_{w}bar"] = (close - ma) / ma * 100

    # -- Realized volatility --
    for w in [15, 30, 60]:
        if len(closes) >= w + 1:
            rets = [(closes[-i] - closes[-i - 1]) / closes[-i - 1]
                    for i in range(1, w + 1)]
            features[f"volatility_{w}bar"] = float(np.std(rets)) * 100

    # -- RSI 14 --
    if len(closes) >= 15:
        gains = [max(closes[-i] - closes[-i - 1], 0) for i in range(1, 15)]
        losses = [max(closes[-i - 1] - closes[-i], 0) for i in range(1, 15)]
        avg_gain = sum(gains) / 14
        avg_loss = sum(losses) / 14
        if avg_loss > 0:
            features["rsi_14"] = 100 - (100 / (1 + avg_gain / avg_loss))
        else:
            features["rsi_14"] = 100.0

    # -- Bollinger band position (20-bar) --
    if len(closes) >= 20:
        bb_ma = sum(closes[-20:]) / 20
        bb_std = float(np.std(closes[-20:]))
        if bb_std > 0:
            features["bb_position"] = (close - (bb_ma - 2 * bb_std)) / (4 * bb_std)
            features["bb_width"] = 4 * bb_std / bb_ma * 100

    # -- Volume ratio --
    if len(volumes) >= 30:
        avg_vol = sum(volumes[-30:]) / 30
        features["volume_ratio"] = volumes[-1] / avg_vol if avg_vol > 0 else 1.0

    # -- High-low range ratio --
    if len(highs) >= 15:
        avg_range = np.mean([
            (highs[-i] - lows[-i]) / closes[-i] * 100
            for i in range(1, 16)
        ])
        current_range = (rec["high"] - rec["low"]) / close * 100
        features["range_ratio"] = current_range / avg_range if avg_range > 0 else 1.0

    # -- Multi-timescale Hurst --
    # Use 60+ bar windows as recommended (30-bar Hurst is too noisy on 1-min)
    if len(closes) >= 60:
        features["hurst_60bar"] = estimate_hurst(closes[-60:], min_window=8)

    if len(closes) >= 120:
        features["hurst_120bar"] = estimate_hurst(closes[-120:], min_window=10)

    # Hurst divergence: short vs long timescale
    if "hurst_60bar" in features and "hurst_120bar" in features:
        features["hurst_divergence"] = features["hurst_60bar"] - features["hurst_120bar"]

    # Regime strength
    if "hurst_60bar" in features:
        h = features["hurst_60bar"]
        features["hurst_regime_strength"] = abs(h - 0.5) * 2
        features["hurst_regime_direction"] = (h - 0.5) * 2

    # -- Rolling z-scores at multiple windows --
    for w in [20, 60, 120]:
        if len(closes) >= w:
            features[f"zscore_{w}bar"] = compute_rolling_zscore(closes, window=w)

    return features


# ===================================================================
# SEQUENCE BUILDER
# ===================================================================

def build_intrabar_sequences(records):
    """Build sequences with multi-horizon return labels.

    Input channels: [close, high, low, volume, hurst_60bar] (5 channels)
    Sequence length: 60 bars (1 hour)

    Returns:
        sequences: (N, 60, 5) input arrays
        returns: (N, len(MULTI_HORIZONS)) actual future returns per horizon
        datetimes: list of datetime strings
    """
    sequences = []
    returns = []
    datetimes = []

    max_horizon = max(MULTI_HORIZONS)

    # Precompute rolling Hurst (60-bar window, updated every bar)
    closes_all = [r["close"] for r in records]
    rolling_hurst = []
    for i in range(len(records)):
        if i >= 60:
            h = estimate_hurst(closes_all[i - 60:i + 1], min_window=8)
            rolling_hurst.append(h)
        else:
            rolling_hurst.append(0.5)

    # Input channels: close, high, low, volume, hurst
    for i in range(SEQUENCE_LENGTH, len(records) - max_horizon):
        seq = []
        for j in range(i - SEQUENCE_LENGTH, i):
            row = [
                float(records[j]["close"]),
                float(records[j]["high"]),
                float(records[j]["low"]),
                float(records[j]["volume"]),
                rolling_hurst[j],
            ]
            seq.append(row)

        last_close = records[i - 1]["close"]
        seq_arr = np.array(seq, dtype=np.float32)

        # Normalize price columns by last close
        for ci in range(3):  # close, high, low
            seq_arr[:, ci] = seq_arr[:, ci] / last_close - 1.0

        # Normalize volume by mean
        vol_mean = seq_arr[:, 3].mean()
        if vol_mean > 0:
            seq_arr[:, 3] = seq_arr[:, 3] / vol_mean - 1.0

        # Hurst: center around 0.5 and scale
        seq_arr[:, 4] = (seq_arr[:, 4] - 0.5) * 4  # Range ~[-2, 2]

        sequences.append(seq_arr)

        # Multi-horizon returns (percentage)
        current_close = records[i - 1]["close"]
        horizon_returns = []
        for h in MULTI_HORIZONS:
            future_close = records[i + h - 1]["close"]
            ret = (future_close - current_close) / current_close * 100
            horizon_returns.append(ret)
        returns.append(horizon_returns)
        datetimes.append(records[i - 1]["datetime"])

    return np.array(sequences), np.array(returns, dtype=np.float32), datetimes


# ===================================================================
# MODEL
# ===================================================================

def _get_model_class():
    """Return the IntrabarQuantileLSTM class (requires torch)."""
    import torch
    import torch.nn as nn

    class IntrabarQuantileLSTM(nn.Module):
        """Shared LSTM backbone + per-horizon quantile heads.

        Same architecture as daily MultiHorizonQuantileLSTM but with
        input_size=5 (no F&G or funding at 1-min resolution).
        """
        def __init__(self, input_size=5, hidden_size=96, num_layers=2,
                     dropout=0.3, n_horizons=5, n_quantiles=5):
            super().__init__()
            self.n_horizons = n_horizons
            self.n_quantiles = n_quantiles

            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, dropout=dropout)

            self.shared = nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
            )

            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, n_quantiles),
                )
                for _ in range(n_horizons)
            ])

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            last_hidden = lstm_out[:, -1, :]
            shared = self.shared(last_hidden)
            outputs = [head(shared) for head in self.heads]
            return torch.stack(outputs, dim=1)  # (batch, n_horizons, n_quantiles)

    return IntrabarQuantileLSTM


def _pinball_loss(predictions, targets, quantiles_tensor):
    """Quantile regression (pinball) loss.

    predictions: (batch, n_horizons, n_quantiles)
    targets: (batch, n_horizons)
    quantiles_tensor: (n_quantiles,)
    """
    import torch
    targets_expanded = targets.unsqueeze(-1)  # (batch, n_horizons, 1)
    errors = targets_expanded - predictions   # (batch, n_horizons, n_quantiles)
    loss = torch.max(
        quantiles_tensor * errors,
        (quantiles_tensor - 1) * errors
    )
    return loss.mean()


# ===================================================================
# TRAINING -- LSTM
# ===================================================================

def train_intrabar_lstm(sequences, returns, asset):
    """Train intrabar multi-horizon quantile LSTM.

    Pinball loss, early stopping, ReduceLROnPlateau.
    Reports directional accuracy and p10-p90 coverage per horizon.
    """
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("    PyTorch not installed. pip install torch")
        return None

    # Chunk size for CPU-friendly validation-set forward passes.
    # At 10K+ test sequences a single model(X_test) call allocates ~480 MB
    # of transient LSTM activations and hits BLAS thread contention.
    TEST_CHUNK = 1024

    n_horizons = len(MULTI_HORIZONS)
    n_quantiles = len(QUANTILES)

    split = int(len(sequences) * TRAIN_SPLIT)
    X_train = torch.FloatTensor(sequences[:split])
    y_train = torch.FloatTensor(returns[:split])
    X_test = torch.FloatTensor(sequences[split:])
    y_test = torch.FloatTensor(returns[split:])

    print(f"    Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"    Horizons: {MULTI_HORIZONS} | Quantiles: {QUANTILES}")
    print(f"    Output: {n_horizons} horizons x {n_quantiles} quantiles "
          f"= {n_horizons * n_quantiles} outputs")

    IntrabarQuantileLSTM = _get_model_class()
    input_size = X_train.shape[2]
    model = IntrabarQuantileLSTM(input_size, n_horizons=n_horizons,
                                  n_quantiles=n_quantiles)
    quantiles_tensor = torch.FloatTensor(QUANTILES)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8)

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    best_test_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    epochs = 150

    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = _pinball_loss(pred, batch_y, quantiles_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            # Chunked forward to keep peak activation memory bounded.
            # Unchunked at 10K+ test sequences this balloons to ~480 MB
            # transient allocation and hits MKL-BLAS thread contention (v5).
            test_preds = []
            for start in range(0, len(X_test), TEST_CHUNK):
                test_preds.append(model(X_test[start:start + TEST_CHUNK]))
            test_pred = torch.cat(test_preds, dim=0)
            test_loss = _pinball_loss(test_pred, y_test, quantiles_tensor).item()

            median_idx = QUANTILES.index(0.50)
            median_preds = test_pred[:, :, median_idx]
            direction_correct = ((median_preds > 0) == (y_test > 0)).float().mean(dim=0)

        scheduler.step(test_loss)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch < 10 or (epoch + 1) % 5 == 0:
            dir_strs = " ".join(f"{MULTI_HORIZONS[h]}bar={direction_correct[h]:.0%}"
                                for h in range(n_horizons))
            elapsed = time.time() - t0
            print(f"    Epoch {epoch+1:>3d}: loss={total_loss/len(loader):.4f} "
                  f"test_loss={test_loss:.4f} dir=[{dir_strs}] "
                  f"({elapsed:.0f}s)", flush=True)

        if patience_counter >= 20:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    # Final evaluation with best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        final_preds = []
        for start in range(0, len(X_test), TEST_CHUNK):
            final_preds.append(model(X_test[start:start + TEST_CHUNK]))
        final_pred = torch.cat(final_preds, dim=0)
        median_idx = QUANTILES.index(0.50)

        print(f"\n    Final test results:")
        for hi, h in enumerate(MULTI_HORIZONS):
            median_preds = final_pred[:, hi, median_idx]
            actual = y_test[:, hi]
            dir_acc = ((median_preds > 0) == (actual > 0)).float().mean()

            p10 = final_pred[:, hi, 0]
            p90 = final_pred[:, hi, -1]
            coverage = ((actual >= p10) & (actual <= p90)).float().mean()

            mae = (median_preds - actual).abs().mean()

            print(f"      {h:>2d}bar: direction={dir_acc:.1%} | "
                  f"p10-p90 coverage={coverage:.1%} (ideal=80%) | "
                  f"MAE={mae:.4f}%")

    elapsed_total = time.time() - t0
    print(f"    Training time: {elapsed_total:.0f}s")

    return {
        "model_state": best_model_state,
        "input_size": input_size,
        "n_horizons": n_horizons,
        "n_quantiles": n_quantiles,
        "horizons": MULTI_HORIZONS,
        "quantiles": QUANTILES,
        "test_loss": float(best_test_loss),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }


# ===================================================================
# TRAINING -- XGBOOST
# ===================================================================

def train_intrabar_xgboost(feature_rows, asset, horizons=None):
    """Train XGBoost quantile regression per horizon.

    Provides a second opinion on distribution shape alongside the LSTM.
    """
    if horizons is None:
        horizons = MULTI_HORIZONS

    results = {}

    for horizon in horizons:
        target_col = f"target_{horizon}bar_return"
        valid = [f for f in feature_rows if target_col in f]

        if len(valid) < 200:
            print(f"    {horizon}bar: Only {len(valid)} rows, need 200+")
            continue

        feat_cols = sorted([k for k in valid[0] if not k.startswith("target_")
                            and k not in ("datetime", "close")
                            and isinstance(valid[0][k], (int, float))])

        X = np.array([[f.get(c, 0) for c in feat_cols] for f in valid])
        y = np.array([f[target_col] for f in valid])
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        split = int(len(X) * TRAIN_SPLIT)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        print(f"\n    {horizon}bar Quantile Regression: "
              f"train={len(X_train)} test={len(X_test)}")

        try:
            import xgboost as xgb

            quantile_models = {}
            for q in QUANTILES:
                model = xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.7,
                    min_child_weight=5,
                    objective="reg:quantileerror",
                    quantile_alpha=q,
                    random_state=42,
                )
                model.fit(X_train, y_train,
                          eval_set=[(X_test, y_test)],
                          verbose=False)
                quantile_models[q] = model

            # Evaluate
            test_preds = {q: m.predict(X_test) for q, m in quantile_models.items()}
            median_pred = test_preds[0.50]
            dir_acc = ((median_pred > 0) == (y_test > 0)).mean()

            coverage = ((y_test >= test_preds[0.10]) &
                        (y_test <= test_preds[0.90])).mean()

            mae = np.abs(median_pred - y_test).mean()

            print(f"      direction={dir_acc:.1%} | coverage={coverage:.1%} "
                  f"| MAE={mae:.4f}%")

            # Top features from median model
            importances = quantile_models[0.50].feature_importances_
            top = sorted(zip(feat_cols, importances), key=lambda x: -x[1])[:5]
            print(f"      Top: {', '.join(f'{n}={v:.3f}' for n, v in top)}")

            results[horizon] = {
                "models": quantile_models,
                "feature_cols": feat_cols,
                "dir_accuracy": float(dir_acc),
                "coverage": float(coverage),
                "mae": float(mae),
            }

        except Exception as e:
            print(f"      Error: {e}")

    return results


# ===================================================================
# TRIPLE BARRIER LABELING (v8.1)
# ===================================================================

def _compute_atr_np(highs, lows, closes, period=14):
    """Average True Range, pure-numpy rolling mean (no pandas)."""
    highs = np.asarray(highs, dtype=np.float64)
    lows = np.asarray(lows, dtype=np.float64)
    closes = np.asarray(closes, dtype=np.float64)
    prev_close = np.concatenate([[closes[0]], closes[:-1]])

    tr1 = highs - lows
    tr2 = np.abs(highs - prev_close)
    tr3 = np.abs(lows - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))

    # Rolling mean via cumulative-sum trick
    atr = np.full(len(tr), np.nan)
    if len(tr) >= period:
        csum = np.cumsum(tr, dtype=np.float64)
        atr[period - 1] = csum[period - 1] / period
        atr[period:] = (csum[period:] - csum[:-period]) / period
    return atr


def _compute_triple_barrier_labels(close_prices, atr_series,
                                   lookforward=15, atr_multiplier=1.5):
    """Triple-barrier labels (+1 UP / -1 DOWN / 0 TIMEOUT) using close-only.

    Returns an array the same length as close_prices. The last `lookforward`
    bars and any bar whose ATR is NaN return np.nan.
    """
    close_prices = np.asarray(close_prices, dtype=np.float64)
    atr_series = np.asarray(atr_series, dtype=np.float64)
    n = len(close_prices)
    labels = np.full(n, np.nan)

    for i in range(n - lookforward):
        atr_i = atr_series[i]
        if not np.isfinite(atr_i) or atr_i <= 0:
            continue

        entry = close_prices[i]
        barrier = atr_multiplier * atr_i
        upper = entry + barrier
        lower = entry - barrier

        future_slice = close_prices[i + 1:i + 1 + lookforward]
        upper_hits = np.where(future_slice >= upper)[0]
        lower_hits = np.where(future_slice <= lower)[0]

        if len(upper_hits) == 0 and len(lower_hits) == 0:
            labels[i] = 0
        elif len(upper_hits) == 0:
            labels[i] = -1
        elif len(lower_hits) == 0:
            labels[i] = 1
        else:
            labels[i] = 1 if upper_hits[0] < lower_hits[0] else -1

    return labels


def train_intrabar_classifier(data, asset, lookforward=15,
                              atr_multiplier=1.5, model_type="xgb"):
    """Train a triple-barrier classifier on the tabular feature set.

    `data` is the dict loaded from {asset}_features.joblib
    (must contain "feature_rows" and "records").

    Returns a dict with metrics + artifacts, or None on hard failure.
    """
    feature_rows = data.get("feature_rows", [])
    records = data.get("records", [])
    if not feature_rows or not records:
        print("    Missing feature_rows or records in features file")
        return None

    highs = np.array([r["high"] for r in records], dtype=np.float64)
    lows = np.array([r["low"] for r in records], dtype=np.float64)
    closes_all = np.array([r["close"] for r in records], dtype=np.float64)

    atr_all = _compute_atr_np(highs, lows, closes_all, period=14)
    labels_all = _compute_triple_barrier_labels(
        closes_all, atr_all,
        lookforward=lookforward, atr_multiplier=atr_multiplier,
    )

    # Build label map from datetime -> label so we can align with feature_rows
    # (feature_rows correspond to records[MIN_BARS_FOR_FEATURES:]).
    label_by_dt = {}
    for i, rec in enumerate(records):
        if np.isfinite(labels_all[i]):
            label_by_dt[rec["datetime"]] = int(labels_all[i])

    feat_cols = sorted([k for k in feature_rows[0]
                        if not k.startswith("target_")
                        and k not in ("datetime", "close")
                        and isinstance(feature_rows[0][k], (int, float))])

    X_list, y_list = [], []
    for f in feature_rows:
        dt = f.get("datetime")
        if dt in label_by_dt:
            X_list.append([f.get(c, 0) for c in feat_cols])
            y_list.append(label_by_dt[dt])

    if not X_list:
        print("    No labeled feature rows aligned")
        return None

    X = np.nan_to_num(np.asarray(X_list, dtype=np.float64),
                      nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray(y_list, dtype=np.int64)

    n_total = len(y)
    n_up = int((y == 1).sum())
    n_dn = int((y == -1).sum())
    n_to = int((y == 0).sum())
    print(f"    Labeled rows: {n_total}")
    print(f"    Class distribution:")
    print(f"      UP      (+1): {n_up:>6d} ({n_up/n_total:.1%})")
    print(f"      DOWN    (-1): {n_dn:>6d} ({n_dn/n_total:.1%})")
    print(f"      TIMEOUT ( 0): {n_to:>6d} ({n_to/n_total:.1%})")

    split = int(n_total * TRAIN_SPLIT)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"    Train: {len(X_train)} | Test: {len(X_test)}")

    # XGB expects 0..C-1 class labels. Map {-1, 0, 1} -> {0, 1, 2}.
    CLASS_ORDER = [-1, 0, 1]  # indices 0=DOWN, 1=TIMEOUT, 2=UP
    to_idx = {-1: 0, 0: 1, 1: 2}
    y_train_idx = np.vectorize(to_idx.get)(y_train).astype(np.int64)
    y_test_idx = np.vectorize(to_idx.get)(y_test).astype(np.int64)

    t0 = time.time()

    if model_type == "xgb":
        try:
            import xgboost as xgb
        except ImportError:
            print("    pip install xgboost")
            return None

        clf = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=5,
            objective="multi:softprob",
            num_class=3,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train_idx,
                eval_set=[(X_test, y_test_idx)],
                verbose=False)
        preds_idx = clf.predict(X_test)
        importances = clf.feature_importances_
        model_artifact = clf
    elif model_type == "mlp":
        try:
            from sklearn.neural_network import MLPClassifier
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            print("    pip install scikit-learn")
            return None
        scaler = StandardScaler().fit(X_train)
        Xt = scaler.transform(X_train)
        Xv = scaler.transform(X_test)
        clf = MLPClassifier(hidden_layer_sizes=(64, 32),
                            max_iter=200, random_state=42)
        clf.fit(Xt, y_train_idx)
        preds_idx = clf.predict(Xv)
        importances = np.zeros(X_train.shape[1])
        model_artifact = (scaler, clf)
    else:
        print(f"    model_type={model_type!r} not implemented in v8.1")
        return None

    elapsed = time.time() - t0

    # Metrics
    test_acc = float((preds_idx == y_test_idx).mean())
    print(f"\n    Training time: {elapsed:.1f}s")
    print(f"    Test accuracy: {test_acc:.3%}")

    # Confusion matrix -- rows=true, cols=predicted, order DOWN/TIMEOUT/UP
    conf = np.zeros((3, 3), dtype=np.int64)
    for t_idx, p_idx in zip(y_test_idx, preds_idx):
        conf[t_idx][p_idx] += 1

    print(f"\n    Confusion matrix (rows=true, cols=pred; "
          f"order DOWN / TIMEOUT / UP):")
    print(f"             pred_DOWN  pred_TIME  pred_UP")
    for i, name in enumerate(["DOWN   ", "TIMEOUT", "UP     "]):
        print(f"      true_{name} {conf[i][0]:>9d}  {conf[i][1]:>9d}  "
              f"{conf[i][2]:>7d}")

    # Per-class precision / recall
    print(f"\n    Per-class metrics:")
    print(f"      {'class':<9s} {'prec':>7s} {'recall':>7s} "
          f"{'support_true':>13s} {'support_pred':>13s}")
    precisions = {}
    recalls = {}
    for i, name in enumerate(["DOWN", "TIMEOUT", "UP"]):
        tp = int(conf[i][i])
        pred_total = int(conf[:, i].sum())
        true_total = int(conf[i, :].sum())
        prec = tp / pred_total if pred_total > 0 else 0.0
        rec = tp / true_total if true_total > 0 else 0.0
        precisions[name] = prec
        recalls[name] = rec
        print(f"      {name:<9s} {prec*100:>6.2f}% {rec*100:>6.2f}% "
              f"{true_total:>13d} {pred_total:>13d}")

    # Tradability metric: precision on UP and DOWN specifically
    print(f"\n    Tradability:")
    print(f"      When model says UP   -> correct {precisions['UP']*100:.2f}% "
          f"of the time ({int(conf[:, 2].sum())} predictions)")
    print(f"      When model says DOWN -> correct {precisions['DOWN']*100:.2f}% "
          f"of the time ({int(conf[:, 0].sum())} predictions)")

    # Top features (xgb only)
    if model_type == "xgb":
        top = sorted(zip(feat_cols, importances),
                     key=lambda x: -x[1])[:10]
        print(f"\n    Top 10 features by importance:")
        for n, v in top:
            print(f"      {n:<30s} {v:.4f}")

    return {
        "model_type": model_type,
        "model": model_artifact,
        "feature_cols": feat_cols,
        "class_order": CLASS_ORDER,
        "lookforward": lookforward,
        "atr_multiplier": atr_multiplier,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "class_distribution": {"UP": n_up, "DOWN": n_dn, "TIMEOUT": n_to},
        "test_accuracy": test_acc,
        "confusion_matrix": conf.tolist(),
        "precision": precisions,
        "recall": recalls,
        "training_time": elapsed,
    }


# ===================================================================
# CLI COMMANDS
# ===================================================================

def cmd_build_features(args):
    """Build features from N-minute aggregated bars."""
    asset = args.asset.upper()

    print(f"\n{'='*70}")
    print(f"  INTRABAR FEATURE ENGINEERING -- {asset}")
    print(f"  Bar size: {BAR_SIZE_MINUTES} minute(s)")
    print(f"  Horizons: {MULTI_HORIZONS} bars "
          f"({[h * BAR_SIZE_MINUTES for h in MULTI_HORIZONS]} minutes forward)")
    print(f"{'='*70}")

    # Remove stale artifacts from any prior bar-size run
    for old_file in MODEL_DIR.glob(f"{asset}_*"):
        old_file.unlink()
        print(f"  Removed stale: {old_file.name}")

    records = load_intrabar_data(asset)
    if not records:
        return

    if len(records) < MIN_BARS_FOR_FEATURES + max(MULTI_HORIZONS) + 100:
        print(f"  Need at least {MIN_BARS_FOR_FEATURES + max(MULTI_HORIZONS) + 100} "
              f"bars, have {len(records)}")
        return

    # Build tabular features
    feature_rows = []
    skipped = 0
    for i in range(MIN_BARS_FOR_FEATURES, len(records)):
        feats = compute_intrabar_features(records, i)
        if feats:
            feats["datetime"] = records[i]["datetime"]
            feats["close"] = records[i]["close"]

            # Add return targets for each horizon
            for h in MULTI_HORIZONS:
                if i + h < len(records):
                    future = records[i + h]["close"]
                    feats[f"target_{h}bar_return"] = (
                        (future - records[i]["close"]) / records[i]["close"] * 100
                    )

            feature_rows.append(feats)
        else:
            skipped += 1

    print(f"\n  Tabular features: {len(feature_rows)} rows (skipped {skipped})")
    if feature_rows:
        feat_cols = [k for k in feature_rows[0] if not k.startswith("target_")
                     and k not in ("datetime", "close")]
        print(f"  Feature columns ({len(feat_cols)}): "
              f"{', '.join(sorted(feat_cols)[:12])}...")

    # Build LSTM sequences
    seqs, rets, dts = build_intrabar_sequences(records)
    if len(seqs) > 0:
        print(f"  LSTM sequences: {len(seqs)}, shape={seqs[0].shape}")
        print(f"  Return targets shape: {rets.shape}")

        # Quick stats on returns
        for hi, h in enumerate(MULTI_HORIZONS):
            r = rets[:, hi]
            print(f"    {h:>2d}bar: mean={r.mean():+.4f}% "
                  f"std={r.std():.4f}% up_rate={( r > 0).mean():.1%}")

    # Save
    import joblib
    save_path = MODEL_DIR / f"{asset}_features.joblib"
    joblib.dump({
        "feature_rows": feature_rows,
        "records": records,
        "asset": asset,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "n_bars": len(records),
    }, save_path)

    print(f"\n  Saved to {save_path}")
    print(f"{'='*70}")


def cmd_train(args):
    """Train both LSTM and XGBoost multi-horizon models."""
    asset = args.asset.upper()

    print(f"\n{'='*70}")
    print(f"  INTRABAR MULTI-HORIZON TRAINING -- {asset}")
    print(f"  Horizons: {MULTI_HORIZONS} bars")
    print(f"  Quantiles: {QUANTILES}")
    print(f"{'='*70}")

    import joblib
    feat_path = MODEL_DIR / f"{asset}_features.joblib"
    if not feat_path.exists():
        print(f"  Run build-features first:")
        print(f"    python -m engines.intrabar_predictor build-features --asset {asset}")
        return

    data = joblib.load(feat_path)
    records = data["records"]
    feature_rows = data["feature_rows"]

    # -- LSTM --
    print(f"\n  {'='*60}")
    print(f"  INTRABAR QUANTILE LSTM")
    print(f"  {'='*60}")

    seqs, returns, dts = build_intrabar_sequences(records)
    print(f"  Sequences: {len(seqs)}, shape={seqs[0].shape}")
    print(f"  Return targets shape: {returns.shape}")

    lstm_result = None
    if len(seqs) > 200:
        lstm_result = train_intrabar_lstm(seqs, returns, asset)

        if lstm_result and lstm_result.get("model_state"):
            try:
                import torch
                torch.save(lstm_result["model_state"],
                           MODEL_DIR / f"{asset}_multi_horizon_lstm.pt")
                print(f"\n  LSTM model saved")
            except Exception as e:
                print(f"  LSTM save error: {e}")
    else:
        print(f"  Not enough sequences ({len(seqs)}), need 200+")

    # -- XGBOOST --
    print(f"\n  {'='*60}")
    print(f"  INTRABAR QUANTILE XGBOOST")
    print(f"  {'='*60}")

    xgb_result = train_intrabar_xgboost(feature_rows, asset)

    # Save metadata
    save_data = {
        "asset": asset,
        "mode": "intrabar_multi_horizon_quantile",
        "horizons": MULTI_HORIZONS,
        "quantiles": QUANTILES,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "lstm_result": {k: v for k, v in lstm_result.items()
                        if k != "model_state"} if lstm_result else None,
        "xgb_result": {
            h: {k: v for k, v in r.items() if k != "models"}
            for h, r in xgb_result.items()
        } if xgb_result else None,
    }

    if xgb_result:
        joblib.dump(xgb_result, MODEL_DIR / f"{asset}_quant_mh_models.joblib")

    joblib.dump(save_data, MODEL_DIR / f"{asset}_multi_horizon.joblib")
    print(f"\n  Saved: {MODEL_DIR / f'{asset}_multi_horizon.joblib'}")
    print(f"{'='*70}")


def cmd_train_classifier(args):
    """Train triple-barrier classifier on existing features (v8.1).

    Adds alongside the existing LSTM quantile-regression `train` path.
    Saves to {asset}_classifier_{model}.joblib to avoid clobbering.
    """
    asset = args.asset.upper()
    lookforward = args.lookforward
    atr_mult = args.atr_mult
    model_type = args.model

    print(f"\n{'='*70}")
    print(f"  TRIPLE-BARRIER CLASSIFIER TRAINING -- {asset}")
    print(f"  Lookforward: {lookforward} bars "
          f"({lookforward * BAR_SIZE_MINUTES} min)")
    print(f"  ATR multiplier: {atr_mult}  (barrier = {atr_mult} x ATR14)")
    print(f"  Model: {model_type}")
    print(f"{'='*70}")

    import joblib
    feat_path = MODEL_DIR / f"{asset}_features.joblib"
    if not feat_path.exists():
        print(f"  No features at {feat_path}. Run build-features first.")
        return

    data = joblib.load(feat_path)
    print(f"  Loaded {feat_path.name}")

    result = train_intrabar_classifier(
        data, asset,
        lookforward=lookforward,
        atr_multiplier=atr_mult,
        model_type=model_type,
    )
    if result is None:
        return

    save_path = MODEL_DIR / f"{asset}_classifier_{model_type}.joblib"
    joblib.dump(result, save_path)
    print(f"\n  Saved: {save_path}")
    print(f"{'='*70}")


def cmd_xgb_only_probe(args):
    """Train XGBoost quantile ensemble only. No LSTM. Signal-presence probe.

    Loads existing {asset}_features.joblib, trains the same XGB ensemble
    used by cmd_train, and saves to {asset}_xgb_probe.joblib so existing
    LSTM-paired artifacts (quant_mh_models.joblib, multi_horizon.joblib,
    multi_horizon_lstm.pt) are not touched.
    """
    asset = args.asset.upper()

    print(f"\n{'='*70}")
    print(f"  XGB-ONLY SIGNAL PROBE -- {asset}")
    print(f"  Horizons: {MULTI_HORIZONS} bars")
    print(f"  Quantiles: {QUANTILES}")
    print(f"{'='*70}")

    import joblib
    feat_path = MODEL_DIR / f"{asset}_features.joblib"
    if not feat_path.exists():
        print(f"  No features at {feat_path}. Run build-features first.")
        return

    data = joblib.load(feat_path)
    feature_rows = data.get("feature_rows", [])
    if not feature_rows:
        print(f"  feature_rows empty in {feat_path}")
        return

    print(f"  Loaded {len(feature_rows)} feature rows from {feat_path.name}")

    t0 = time.time()
    xgb_result = train_intrabar_xgboost(feature_rows, asset)
    elapsed = time.time() - t0

    if not xgb_result:
        print(f"\n  XGB probe produced no results.")
        return

    save_path = MODEL_DIR / f"{asset}_xgb_probe.joblib"
    joblib.dump(xgb_result, save_path)

    print(f"\n  {'='*60}")
    print(f"  XGB-ONLY PROBE SUMMARY -- {asset}")
    print(f"  {'='*60}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  {'Horizon':<10s} {'dir_acc':>10s} {'coverage':>10s} {'MAE':>10s}")
    for h in MULTI_HORIZONS:
        if h in xgb_result:
            r = xgb_result[h]
            print(f"  {h:>2d}bar     {r['dir_accuracy']*100:>9.2f}% "
                  f"{r['coverage']*100:>9.2f}% {r['mae']:>9.4f}%")
        else:
            print(f"  {h:>2d}bar     (no result)")

    print(f"\n  Saved: {save_path}")
    print(f"{'='*70}")


def cmd_predict(args):
    """Generate intrabar predictions for current state."""
    asset = args.asset.upper()

    print(f"\n{'='*70}")
    print(f"  INTRABAR PREDICTION -- {asset}")
    print(f"{'='*70}")

    import joblib

    records = load_intrabar_data(asset)
    if not records:
        return

    current_price = records[-1]["close"]
    print(f"  Current price: ${current_price:,.2f}")
    print(f"  Last bar: {records[-1]['datetime']}")

    # Load model metadata
    mh_path = MODEL_DIR / f"{asset}_multi_horizon.joblib"
    if not mh_path.exists():
        print(f"\n  No model. Run: train --asset {asset}")
        return

    saved = joblib.load(mh_path)

    print(f"\n  {'Horizon':<10s} {'p10':>8s} {'p25':>8s} {'p50':>8s} "
          f"{'p75':>8s} {'p90':>8s} {'Dir'}")
    print(f"  {'-'*62}")

    # LSTM prediction
    lstm_path = MODEL_DIR / f"{asset}_multi_horizon_lstm.pt"
    if lstm_path.exists() and saved.get("lstm_result"):
        try:
            import torch
            IntrabarQuantileLSTM = _get_model_class()
            info = saved["lstm_result"]

            model = IntrabarQuantileLSTM(
                info["input_size"],
                n_horizons=info["n_horizons"],
                n_quantiles=info["n_quantiles"])
            model.load_state_dict(torch.load(lstm_path, weights_only=True))
            model.eval()

            seqs, _, _ = build_intrabar_sequences(records)
            if len(seqs) > 0:
                with torch.no_grad():
                    pred = model(torch.FloatTensor(seqs[-1:]))
                    preds = pred[0].numpy()

                for hi, h in enumerate(MULTI_HORIZONS):
                    p = preds[hi]
                    direction = "UP" if p[2] > 0 else "DOWN"
                    print(f"  LSTM {h:>2d}bar {p[0]:>+7.3f}% {p[1]:>+7.3f}% "
                          f"{p[2]:>+7.3f}% {p[3]:>+7.3f}% {p[4]:>+7.3f}%  {direction}")

        except Exception as e:
            print(f"  LSTM error: {e}")

    # XGBoost prediction
    quant_path = MODEL_DIR / f"{asset}_quant_mh_models.joblib"
    if quant_path.exists():
        try:
            quant_models = joblib.load(quant_path)
            feats = compute_intrabar_features(records, len(records) - 1)
            if feats:
                for h, hdata in quant_models.items():
                    feat_cols = hdata["feature_cols"]
                    X = np.array([[feats.get(c, 0) for c in feat_cols]])
                    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

                    preds = {}
                    for q, m in hdata["models"].items():
                        preds[q] = float(m.predict(X)[0])

                    p = [preds[q] for q in QUANTILES]
                    direction = "UP" if preds[0.50] > 0 else "DOWN"
                    print(f"  XGB  {h:>2d}bar {p[0]:>+7.3f}% {p[1]:>+7.3f}% "
                          f"{p[2]:>+7.3f}% {p[3]:>+7.3f}% {p[4]:>+7.3f}%  {direction}")

        except Exception as e:
            print(f"  XGBoost error: {e}")

    print(f"\n{'='*70}")


def cmd_confluence(args):
    """Generate intrabar mean-reversion confluence signals.

    Signal logic:
    1. Compute rolling z-score on 60-bar window
    2. If |z-score| < threshold: no signal
    3. Get LSTM multi-horizon prediction (latest 60 bars)
    4. Check short (5-bar) vs long (15-bar) alignment

    For LONG (z-score <= -threshold, price is cheap):
      - Short-term (5-bar) median > 0 = bounce starting
      - Long-term (15-bar) median > 0 = reversion has room
      - Hurst 60-bar < 0.45 = mean-reverting regime confirmed

    For SHORT (z-score >= +threshold, price is rich):
      - Short-term (5-bar) median < 0 = rollover starting
      - Long-term (15-bar) median < 0 = reversion continuing
    """
    asset = args.asset.upper()
    zscore_threshold = getattr(args, "zscore", 2.0)
    min_magnitude = getattr(args, "min_mag", 0.05)

    # Horizon indices: 5-bar is index 2, 15-bar is index 4
    short_horizon_idx = 2   # 5 bars from MULTI_HORIZONS
    long_horizon_idx = 4    # 15 bars from MULTI_HORIZONS

    print(f"\n{'='*70}")
    print(f"  INTRABAR CONFLUENCE SIGNALS -- {asset}")
    print(f"  Z-score threshold: +/-{zscore_threshold}")
    print(f"  Min predicted magnitude: {min_magnitude}%")
    print(f"  Short horizon: {MULTI_HORIZONS[short_horizon_idx]} bars")
    print(f"  Long horizon: {MULTI_HORIZONS[long_horizon_idx]} bars")
    print(f"{'='*70}")

    import joblib

    records = load_intrabar_data(asset)
    if not records:
        return

    current_price = records[-1]["close"]
    print(f"  Current price: ${current_price:,.2f}")
    print(f"  Last bar: {records[-1]['datetime']}")

    # Compute current z-score (60-bar window)
    closes = [r["close"] for r in records]
    zscore = compute_rolling_zscore(closes, window=60)
    print(f"  Z-score (60-bar): {zscore:+.3f}")

    # Also show 20-bar and 120-bar z-scores for context
    z20 = compute_rolling_zscore(closes, window=20)
    z120 = compute_rolling_zscore(closes, window=120)
    print(f"  Z-score (20-bar): {z20:+.3f}")
    print(f"  Z-score (120-bar): {z120:+.3f}")

    # Compute Hurst
    h60 = 0.5
    if len(closes) >= 60:
        h60 = estimate_hurst(closes[-60:], min_window=8)
        regime = "trending" if h60 > 0.55 else "mean-reverting" if h60 < 0.45 else "random"
        print(f"  Hurst (60-bar): {h60:.3f} ({regime})")

    # Load LSTM model and predict
    mh_path = MODEL_DIR / f"{asset}_multi_horizon.joblib"
    if not mh_path.exists():
        print(f"\n  No model. Run:")
        print(f"    python -m engines.intrabar_predictor build-features --asset {asset}")
        print(f"    python -m engines.intrabar_predictor train --asset {asset}")
        return

    saved = joblib.load(mh_path)

    # Get LSTM predictions
    lstm_preds = None
    lstm_path = MODEL_DIR / f"{asset}_multi_horizon_lstm.pt"
    if lstm_path.exists() and saved.get("lstm_result"):
        try:
            import torch
            IntrabarQuantileLSTM = _get_model_class()
            info = saved["lstm_result"]

            model = IntrabarQuantileLSTM(
                info["input_size"],
                n_horizons=info["n_horizons"],
                n_quantiles=info["n_quantiles"])
            model.load_state_dict(torch.load(lstm_path, weights_only=True))
            model.eval()

            seqs, _, _ = build_intrabar_sequences(records)
            if len(seqs) > 0:
                with torch.no_grad():
                    pred = model(torch.FloatTensor(seqs[-1:]))
                    lstm_preds = pred[0].numpy()

                print(f"\n  Multi-horizon LSTM predictions:")
                print(f"  {'Horizon':<10s} {'p10':>8s} {'p25':>8s} {'p50':>8s} "
                      f"{'p75':>8s} {'p90':>8s} {'Dir'}")
                print(f"  {'-'*62}")
                for hi, h in enumerate(MULTI_HORIZONS):
                    p = lstm_preds[hi]
                    direction = "UP" if p[2] > 0 else "DOWN"
                    print(f"  {h:>2d} bar    {p[0]:>+7.3f}% {p[1]:>+7.3f}% "
                          f"{p[2]:>+7.3f}% {p[3]:>+7.3f}% {p[4]:>+7.3f}%  {direction}")

        except Exception as e:
            print(f"  LSTM prediction error: {e}")

    # -- CONFLUENCE CHECK --
    print(f"\n  {'='*60}")
    print(f"  CONFLUENCE ANALYSIS")
    print(f"  {'='*60}")

    signals = []

    if abs(zscore) < zscore_threshold:
        print(f"\n  Z-score {zscore:+.3f} is within +/-{zscore_threshold}")
        print(f"  No mean-reversion signal active. Price is not extended.")
    else:
        direction = "LONG" if zscore < 0 else "SHORT"
        print(f"\n  Z-score {zscore:+.3f} exceeds threshold -> {direction} candidate")

        if lstm_preds is not None:
            median_idx = QUANTILES.index(0.50)
            short_median = lstm_preds[short_horizon_idx][median_idx]
            long_median = lstm_preds[long_horizon_idx][median_idx]

            short_h = MULTI_HORIZONS[short_horizon_idx]
            long_h = MULTI_HORIZONS[long_horizon_idx]

            print(f"\n  LSTM dual-horizon check:")
            print(f"    {short_h}-bar median: {short_median:+.4f}%")
            print(f"    {long_h}-bar median: {long_median:+.4f}%")

            # Minimum magnitude filter
            short_mag_ok = abs(short_median) >= min_magnitude
            long_mag_ok = abs(long_median) >= min_magnitude

            if not short_mag_ok:
                print(f"    {short_h}-bar predicted move ({abs(short_median):.4f}%) "
                      f"below minimum {min_magnitude}% -- too small to trade")
            if not long_mag_ok:
                print(f"    {long_h}-bar predicted move ({abs(long_median):.4f}%) "
                      f"below minimum {min_magnitude}% -- too small to trade")

            if direction == "LONG":
                short_ok = short_median > 0 and short_mag_ok
                long_ok = long_median > 0 and long_mag_ok
                confluence = short_ok and long_ok

                print(f"    Short-term UP (bounce starting): "
                      f"{'YES' if short_ok else 'NO'}")
                print(f"    Long-term UP (room to revert):   "
                      f"{'YES' if long_ok else 'NO'}")

            else:  # SHORT
                short_ok = short_median < 0 and short_mag_ok
                long_ok = long_median < 0 and long_mag_ok
                confluence = short_ok and long_ok

                print(f"    Short-term DOWN (rollover):      "
                      f"{'YES' if short_ok else 'NO'}")
                print(f"    Long-term DOWN (reversion):      "
                      f"{'YES' if long_ok else 'NO'}")

            hurst_ok = h60 < 0.45

            if confluence and hurst_ok:
                print(f"\n    >>> TRIPLE CONFLUENCE: Z-score + Dual-horizon + Hurst <<<")
                print(f"    >>> STRONG {direction} SIGNAL")
                signals.append({
                    "type": direction, "strength": "STRONG",
                    "zscore": zscore, "short": short_median,
                    "long": long_median, "hurst": h60,
                })
            elif confluence:
                print(f"\n    >>> DUAL CONFLUENCE: Z-score + Dual-horizon <<<")
                print(f"    >>> MODERATE {direction} SIGNAL")
                signals.append({
                    "type": direction, "strength": "MODERATE",
                    "zscore": zscore, "short": short_median,
                    "long": long_median,
                })
            else:
                print(f"\n    No confluence. Horizons don't align for {direction}.")
                if direction == "LONG" and not short_ok:
                    print(f"    Short-term still predicts DOWN -- wait for bounce")
                elif direction == "SHORT" and not short_ok:
                    print(f"    Short-term still predicts UP -- wait for rollover")
        else:
            print(f"  No LSTM predictions available for confluence check.")

    if not signals:
        print(f"\n  No actionable confluence signals right now.")
    else:
        print(f"\n  Active signals: {len(signals)}")
        for s in signals:
            print(f"    {s['strength']} {s['type']}: z={s['zscore']:+.3f} "
                  f"short={s['short']:+.4f}% long={s['long']:+.4f}%")

    print(f"\n{'='*70}")


def cmd_backtest(args):
    """Walk-forward backtest of confluence signals.

    For each historical bar:
    1. Check if z-score threshold hit
    2. If yes, run LSTM prediction on preceding 60 bars
    3. Check dual-horizon confluence
    4. If signal, record hypothetical entry and track for 15 bars forward
    5. Compute hit rate, MAE, P&L distribution

    Walk-forward only -- no peeking at future data.
    Reports results with and without transaction costs.
    """
    asset = args.asset.upper()
    zscore_threshold = getattr(args, "zscore", 2.0)
    min_magnitude = getattr(args, "min_mag", 0.05)
    fees_mode = getattr(args, "fees_mode", "default")
    fees_custom = getattr(args, "fees_custom", None)
    regime_split = getattr(args, "regime_split", False)

    # Resolve per-side fee (fractional units, e.g. 0.0015 = 0.15%)
    if fees_mode == "custom":
        if fees_custom is None:
            print("  --fees-mode=custom requires --fees-custom <float>")
            return
        fee_per_side = fees_custom
    else:
        fee_per_side = FEE_PRESETS[fees_mode]
    round_trip = fee_per_side * 2
    # Convert to percentage-point units to match raw_pnl (which is in % terms)
    fee_per_side_pct = fee_per_side * 100
    round_trip_pct = round_trip * 100

    print(f"\n{'='*70}")
    print(f"  INTRABAR WALK-FORWARD BACKTEST -- {asset}")
    print(f"  Z-score threshold: +/-{zscore_threshold}")
    print(f"  Min magnitude: {min_magnitude}%")
    print(f"  Fee mode: {fees_mode} | per-side: {fee_per_side_pct:.3f}% | "
          f"round-trip: {round_trip_pct:.3f}%")
    if regime_split:
        print(f"  Regime split: ON "
              f"(reverting H<{HURST_REVERTING}, trending H>={HURST_TRENDING})")
    print(f"{'='*70}")

    import joblib

    # Load model
    mh_path = MODEL_DIR / f"{asset}_multi_horizon.joblib"
    lstm_path = MODEL_DIR / f"{asset}_multi_horizon_lstm.pt"
    if not mh_path.exists() or not lstm_path.exists():
        print(f"  No model. Run train first.")
        return

    saved = joblib.load(mh_path)
    if not saved.get("lstm_result"):
        print(f"  No LSTM result in saved model.")
        return

    try:
        import torch
    except ImportError:
        print(f"  PyTorch required for backtest.")
        return

    IntrabarQuantileLSTM = _get_model_class()
    info = saved["lstm_result"]
    model = IntrabarQuantileLSTM(
        info["input_size"],
        n_horizons=info["n_horizons"],
        n_quantiles=info["n_quantiles"])
    model.load_state_dict(torch.load(lstm_path, weights_only=True))
    model.eval()

    # Build all sequences (we'll iterate over the test portion)
    feat_path = MODEL_DIR / f"{asset}_features.joblib"
    if not feat_path.exists():
        print(f"  No features. Run build-features first.")
        return

    data = joblib.load(feat_path)
    records = data["records"]
    feature_rows = data.get("feature_rows", [])

    # Map rec_idx -> feature row so we can read Hurst at signal time
    # without recomputing. feature_rows[k] corresponds to records[MIN_BARS_FOR_FEATURES + k]
    # (build-features is dense with skipped=0 for our data).
    feat_by_rec_idx = {}
    for k, row in enumerate(feature_rows):
        feat_by_rec_idx[MIN_BARS_FOR_FEATURES + k] = row

    seqs, rets, dts = build_intrabar_sequences(records)
    if len(seqs) == 0:
        print(f"  No sequences built.")
        return

    # Only backtest on test portion (last 20%)
    split = int(len(seqs) * TRAIN_SPLIT)
    test_start = split

    print(f"  Total sequences: {len(seqs)}")
    print(f"  Backtesting on test set: indices {test_start} to {len(seqs) - 1}")
    print(f"  Test bars: {dts[test_start]} to {dts[-1]}")

    # Precompute z-scores for all bars
    closes_all = [r["close"] for r in records]
    median_idx = QUANTILES.index(0.50)
    short_horizon_idx = 2  # 5-bar
    long_horizon_idx = 4   # 15-bar
    exit_horizon = 15      # Hold for 15 bars then close

    trades = []
    total_signals_checked = 0

    for i in range(test_start, len(seqs)):
        # Map sequence index back to records index
        rec_idx = i + SEQUENCE_LENGTH  # Sequence i starts at SEQUENCE_LENGTH offset

        if rec_idx + exit_horizon >= len(records):
            break

        # Z-score at this bar
        bar_closes = closes_all[max(0, rec_idx - 60):rec_idx + 1]
        z = compute_rolling_zscore(bar_closes, window=60)

        if abs(z) < zscore_threshold:
            continue

        total_signals_checked += 1
        direction = "LONG" if z < 0 else "SHORT"

        # LSTM prediction for this bar
        with torch.no_grad():
            pred = model(torch.FloatTensor(seqs[i:i + 1]))
            preds = pred[0].numpy()

        short_median = preds[short_horizon_idx][median_idx]
        long_median = preds[long_horizon_idx][median_idx]

        # Magnitude filter
        if abs(short_median) < min_magnitude or abs(long_median) < min_magnitude:
            continue

        # Confluence check
        if direction == "LONG":
            if short_median <= 0 or long_median <= 0:
                continue
        else:
            if short_median >= 0 or long_median >= 0:
                continue

        # Read Hurst from the feature row computed at signal time
        # (walk-forward safe -- feature was built using only records[:rec_idx+1])
        feat_row = feat_by_rec_idx.get(rec_idx)
        if feat_row is not None and "hurst_60bar" in feat_row:
            h60 = feat_row["hurst_60bar"]
        else:
            # Fallback for edge cases where feature row is missing
            hurst_closes = closes_all[max(0, rec_idx - 60):rec_idx + 1]
            h60 = (estimate_hurst(hurst_closes, min_window=8)
                   if len(hurst_closes) >= 60 else 0.5)

        hurst_ok = h60 < HURST_REVERTING
        strength = "STRONG" if hurst_ok else "MODERATE"

        # Regime classification at entry
        if h60 < HURST_REVERTING:
            regime = "reverting"
        elif h60 >= HURST_TRENDING:
            regime = "trending"
        else:
            regime = "random"

        # Record trade
        entry_price = records[rec_idx]["close"]
        exit_price = records[rec_idx + exit_horizon]["close"]

        if direction == "LONG":
            raw_pnl = (exit_price - entry_price) / entry_price * 100
        else:
            raw_pnl = (entry_price - exit_price) / entry_price * 100

        net_pnl = raw_pnl - round_trip_pct

        trades.append({
            "datetime": dts[i],
            "direction": direction,
            "strength": strength,
            "regime": regime,
            "zscore": z,
            "short_pred": short_median,
            "long_pred": long_median,
            "hurst": h60,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "raw_pnl_pct": raw_pnl,
            "net_pnl_pct": net_pnl,
            "won": raw_pnl > 0,
        })

    # Report
    print(f"\n  {'='*60}")
    print(f"  BACKTEST RESULTS")
    print(f"  {'='*60}")

    print(f"  Z-score threshold triggers: {total_signals_checked}")
    print(f"  Confluence signals: {len(trades)}")
    filter_rate = (1 - len(trades) / total_signals_checked * 100) if total_signals_checked > 0 else 0
    if total_signals_checked > 0:
        print(f"  Filter rate: {(1 - len(trades)/total_signals_checked):.0%} "
              f"of z-score triggers rejected")

    if not trades:
        print(f"\n  No confluence trades in test period.")
        print(f"  Try lowering --zscore threshold (current: {zscore_threshold})")
        print(f"{'='*70}")
        return

    raw_pnls = [t["raw_pnl_pct"] for t in trades]
    net_pnls = [t["net_pnl_pct"] for t in trades]
    wins = sum(1 for t in trades if t["won"])
    longs = sum(1 for t in trades if t["direction"] == "LONG")
    shorts = len(trades) - longs
    strong = sum(1 for t in trades if t["strength"] == "STRONG")

    print(f"\n  Total trades: {len(trades)} ({longs} long, {shorts} short)")
    print(f"  Triple confluence (strong): {strong}")
    print(f"  Dual confluence (moderate): {len(trades) - strong}")

    print(f"\n  -- Without costs --")
    print(f"  Hit rate: {wins}/{len(trades)} = {wins/len(trades):.1%}")
    print(f"  Mean P&L: {np.mean(raw_pnls):+.4f}%")
    print(f"  Median P&L: {np.median(raw_pnls):+.4f}%")
    print(f"  Std P&L: {np.std(raw_pnls):.4f}%")
    print(f"  Best: {max(raw_pnls):+.4f}% | Worst: {min(raw_pnls):+.4f}%")
    print(f"  Total: {sum(raw_pnls):+.4f}%")

    print(f"\n  -- With costs ({fee_per_side_pct:.3f}% per side, "
          f"{round_trip_pct:.3f}% round-trip) --")
    net_wins = sum(1 for p in net_pnls if p > 0)
    print(f"  Hit rate: {net_wins}/{len(trades)} = {net_wins/len(trades):.1%}")
    print(f"  Mean P&L: {np.mean(net_pnls):+.4f}%")
    print(f"  Total: {sum(net_pnls):+.4f}%")

    # Win rate by strength
    if strong > 0:
        strong_trades = [t for t in trades if t["strength"] == "STRONG"]
        strong_wins = sum(1 for t in strong_trades if t["won"])
        strong_mean = np.mean([t["raw_pnl_pct"] for t in strong_trades])
        print(f"\n  -- Strong signals only --")
        print(f"  Hit rate: {strong_wins}/{strong} = {strong_wins/strong:.1%}")
        print(f"  Mean P&L: {strong_mean:+.4f}%")

    moderate_trades = [t for t in trades if t["strength"] == "MODERATE"]
    if moderate_trades:
        mod_wins = sum(1 for t in moderate_trades if t["won"])
        mod_mean = np.mean([t["raw_pnl_pct"] for t in moderate_trades])
        print(f"\n  -- Moderate signals only --")
        print(f"  Hit rate: {mod_wins}/{len(moderate_trades)} "
              f"= {mod_wins/len(moderate_trades):.1%}")
        print(f"  Mean P&L: {mod_mean:+.4f}%")

    # Equity curve (cumulative net P&L)
    cumulative = np.cumsum(net_pnls)
    print(f"\n  -- Equity curve --")
    n_trades = len(trades)
    checkpoints = [0, n_trades // 4, n_trades // 2, 3 * n_trades // 4, n_trades - 1]
    for cp in checkpoints:
        if cp < len(cumulative):
            print(f"    Trade {cp+1:>4d}: cumulative={cumulative[cp]:+.4f}%")

    # Max drawdown
    peak = cumulative[0]
    max_dd = 0
    for val in cumulative:
        if val > peak:
            peak = val
        dd = peak - val
        if dd > max_dd:
            max_dd = dd
    print(f"  Max drawdown: {max_dd:.4f}%")

    # Average R:R (mean win / mean loss)
    wins_pnl = [p for p in raw_pnls if p > 0]
    losses_pnl = [p for p in raw_pnls if p <= 0]
    if wins_pnl and losses_pnl:
        avg_win = np.mean(wins_pnl)
        avg_loss = abs(np.mean(losses_pnl))
        rr = avg_win / avg_loss if avg_loss > 0 else float("inf")
        print(f"  Avg win: {avg_win:+.4f}% | Avg loss: {np.mean(losses_pnl):+.4f}%")
        print(f"  Risk:Reward ratio: {rr:.2f}")

    # Regime-split breakdown
    if regime_split:
        print(f"\n  {'='*60}")
        print(f"  REGIME-SPLIT RESULTS (Hurst at entry time)")
        print(f"  {'='*60}")
        total_check = 0
        header = (f"  {'Regime':<12s} {'Trades':>7s} {'L/S':>7s} "
                  f"{'MeanH':>7s} {'HR(raw)':>9s} {'HR(net)':>9s} "
                  f"{'MeanPnL':>10s} {'CumPnL':>10s}")
        print(header)
        print(f"  {'-'*(len(header)-2)}")
        for reg in ("reverting", "random", "trending"):
            reg_trades = [t for t in trades if t["regime"] == reg]
            n = len(reg_trades)
            total_check += n
            if n == 0:
                print(f"  {reg:<12s} {0:>7d}  (no trades)")
                continue
            r_longs = sum(1 for t in reg_trades if t["direction"] == "LONG")
            r_shorts = n - r_longs
            r_raw = [t["raw_pnl_pct"] for t in reg_trades]
            r_net = [t["net_pnl_pct"] for t in reg_trades]
            hr_raw = sum(1 for p in r_raw if p > 0) / n
            hr_net = sum(1 for p in r_net if p > 0) / n
            mean_h = np.mean([t["hurst"] for t in reg_trades])
            print(f"  {reg:<12s} {n:>7d} {r_longs:>3d}/{r_shorts:<3d} "
                  f"{mean_h:>7.3f} {hr_raw*100:>8.1f}% {hr_net*100:>8.1f}% "
                  f"{np.mean(r_net):>+9.3f}% {sum(r_net):>+9.3f}%")
            if n < 20:
                print(f"    (note: bucket size {n} < 20 -- result not actionable)")
        if total_check != len(trades):
            print(f"  WARN: regime trade counts sum to {total_check}, "
                  f"expected {len(trades)}")

    print(f"\n{'='*70}")


# ===================================================================
# MAIN
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Intrabar Multi-Horizon Confluence Predictor")
    subs = parser.add_subparsers(dest="command")

    p_feat = subs.add_parser("build-features",
                             help="Build features from 1-min data")
    p_feat.add_argument("--asset", required=True)

    p_train = subs.add_parser("train", help="Train LSTM + XGBoost models")
    p_train.add_argument("--asset", required=True)

    p_xgp = subs.add_parser("xgb-only-probe",
                            help="Train XGBoost quantile ensemble only (v7 signal probe)")
    p_xgp.add_argument("--asset", required=True)

    p_tc = subs.add_parser("train-classifier",
                           help="Train triple-barrier classifier (v8.1)")
    p_tc.add_argument("--asset", required=True)
    p_tc.add_argument("--lookforward", type=int, default=15,
                      help="Barrier lookforward in 5-min bars (default 15 = 75 min)")
    p_tc.add_argument("--atr-mult", dest="atr_mult", type=float, default=1.5,
                      help="Barrier distance as ATR multiple (default 1.5)")
    p_tc.add_argument("--model", choices=["xgb", "mlp", "lstm"], default="xgb",
                      help="Classifier type (default xgb)")

    p_pred = subs.add_parser("predict", help="Generate predictions")
    p_pred.add_argument("--asset", required=True)

    p_conf = subs.add_parser("confluence",
                             help="Mean-reversion confluence signals")
    p_conf.add_argument("--asset", required=True)
    p_conf.add_argument("--zscore", type=float, default=2.0)
    p_conf.add_argument("--min-mag", type=float, default=0.05,
                        dest="min_mag",
                        help="Min predicted magnitude %% to trade")

    p_bt = subs.add_parser("backtest",
                           help="Walk-forward backtest of confluence")
    p_bt.add_argument("--asset", required=True)
    p_bt.add_argument("--zscore", type=float, default=2.0)
    p_bt.add_argument("--min-mag", type=float, default=0.05,
                      dest="min_mag",
                      help="Min predicted magnitude %% to trade")
    p_bt.add_argument("--fees-mode", dest="fees_mode",
                      choices=["maker", "taker", "default", "zero", "custom"],
                      default="default",
                      help="Fee preset for cost sensitivity (default: 0.15%% per side)")
    p_bt.add_argument("--fees-custom", dest="fees_custom",
                      type=float, default=None,
                      help="Custom per-side fee as fraction (e.g. 0.0008 = 0.08%%); "
                           "only used when --fees-mode=custom")
    p_bt.add_argument("--regime-split", dest="regime_split",
                      action="store_true",
                      help="Break down backtest results by Hurst regime")

    args = parser.parse_args()

    dispatch = {
        "build-features": cmd_build_features,
        "train": cmd_train,
        "xgb-only-probe": cmd_xgb_only_probe,
        "train-classifier": cmd_train_classifier,
        "predict": cmd_predict,
        "confluence": cmd_confluence,
        "backtest": cmd_backtest,
    }

    if args.command in dispatch:
        dispatch[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()