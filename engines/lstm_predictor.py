"""
engines/lstm_predictor.py -- LSTM + Quantamental Crypto Prediction Ensemble

Two-model ensemble with multi-horizon distributional output:
  1. LSTM: Learns temporal patterns from price/volume sequences
  2. Quantamental: XGBoost on fundamentals, sentiment, on-chain, funding rates

Output modes:
  - Binary: probability of UP vs DOWN (legacy, still supported)
  - Multi-horizon quantile: predicted return distribution at 1d/7d/30d
    with p10/p25/p50/p75/p90 quantiles per horizon
  - Confluence signals: when z-score + near-term + long-term align

The multi-horizon output enables:
  - Mean-reversion timing: z-score threshold + short-term UP + long-term DOWN
  - Polymarket bucket mapping: p90 for "BTC > $X" directly tradeable
  - Risk sizing from prediction intervals (p10 to p90 width)

Usage:
    python -m engines.lstm_predictor build-features --asset BTC
    python -m engines.lstm_predictor train --asset BTC
    python -m engines.lstm_predictor predict --asset BTC
    python -m engines.lstm_predictor confluence --asset BTC   # Mean-reversion signals
    python -m engines.lstm_predictor markets                  # Compare vs Polymarket
    python -m engines.lstm_predictor backtest --asset BTC     # Walk-forward backtest
"""
import argparse
import json
import math
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
load_dotenv()

DATA_DB = Path("data/crypto_data.db")
MODEL_DIR = Path("models/lstm")
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

SUPPORTED_ASSETS = ["BTC", "ETH"]
PREDICTION_HORIZONS = [1, 7, 30]  # Days ahead
SEQUENCE_LENGTH = 60              # 60 days lookback for LSTM
TRAIN_SPLIT = 0.80


# ===================================================================
# DATA LOADING
# ===================================================================

def load_all_data(asset):
    """Load and merge all data sources into a single daily dataframe-like structure."""
    if not DATA_DB.exists():
        print(f"  No data. Run: python -m engines.crypto_data_collector collect-all --asset {asset}")
        return None

    conn = sqlite3.connect(str(DATA_DB))

    # Daily OHLCV
    ohlcv = {}
    for row in conn.execute("""
        SELECT date, open, high, low, close, volume
        FROM ohlcv_daily WHERE asset=? ORDER BY date
    """, (asset,)).fetchall():
        ohlcv[row[0]] = {
            "open": row[1], "high": row[2], "low": row[3],
            "close": row[4], "volume": row[5],
        }

    if len(ohlcv) < 200:
        print(f"  Need 200+ daily candles, have {len(ohlcv)}")
        return None

    # Fear & Greed
    fg = {}
    for row in conn.execute("SELECT date, value FROM fear_greed").fetchall():
        fg[row[0]] = row[1]

    # Funding rates (aggregate to daily)
    funding = {}
    for row in conn.execute("""
        SELECT DATE(datetime) as d, AVG(funding_rate), SUM(funding_rate)
        FROM funding_rates WHERE asset=?
        GROUP BY d
    """, (asset,)).fetchall():
        funding[row[0]] = {"avg_rate": row[1], "cum_rate": row[2]}

    # On-chain (BTC only)
    onchain = {}
    if asset == "BTC":
        for row in conn.execute("""
            SELECT date, active_addresses, transaction_count,
                   hash_rate, difficulty, market_cap
            FROM onchain_btc
        """).fetchall():
            onchain[row[0]] = {
                "active_addresses": row[1], "tx_count": row[2],
                "hash_rate": row[3], "difficulty": row[4],
                "onchain_mcap": row[5],
            }

    conn.close()

    # Merge into daily records
    dates = sorted(ohlcv.keys())
    records = []

    for date in dates:
        rec = {"date": date}
        rec.update(ohlcv[date])
        rec["fear_greed"] = fg.get(date, 50)
        rec["funding_avg"] = funding.get(date, {}).get("avg_rate", 0)
        rec["funding_cum"] = funding.get(date, {}).get("cum_rate", 0)

        if asset == "BTC" and date in onchain:
            rec.update(onchain[date])
        else:
            rec["active_addresses"] = 0
            rec["tx_count"] = 0
            rec["hash_rate"] = 0
            rec["difficulty"] = 0
            rec["onchain_mcap"] = 0

        records.append(rec)

    print(f"  Loaded {len(records)} daily records for {asset}")
    print(f"  Date range: {records[0]['date']} to {records[-1]['date']}")
    print(f"  Fear & Greed coverage: {sum(1 for r in records if r['fear_greed'] != 50)}/{len(records)}")
    print(f"  Funding rate coverage: {sum(1 for r in records if r['funding_avg'] != 0)}/{len(records)}")
    if asset == "BTC":
        print(f"  On-chain coverage: {sum(1 for r in records if r['hash_rate'] > 0)}/{len(records)}")

    return records


# ===================================================================
# FEATURE ENGINEERING
# ===================================================================

def estimate_hurst(prices, min_window=10):
    """Estimate Hurst exponent using Rescaled Range (R/S) analysis.

    The Hurst exponent H controls the memory kernel decay in the
    Volterra framework: K(t,s) ~ (t-s)^(H-1/2)

    H > 0.5: Long memory, trending (momentum regime)
    H = 0.5: No memory, random walk (no edge)
    H < 0.5: Anti-persistent, mean-reverting (reversion regime)

    Uses log-returns, computes R/S statistic at multiple sub-window
    sizes, fits log(R/S) vs log(n) regression. Slope = H.
    """
    n = len(prices)
    if n < min_window * 2:
        return 0.5  # Default to random walk if insufficient data

    # Log returns
    returns = np.diff(np.log(np.array(prices, dtype=np.float64)))
    returns = returns[np.isfinite(returns)]

    if len(returns) < min_window * 2:
        return 0.5

    # Window sizes: powers of 2 from min_window to n//2
    window_sizes = []
    w = min_window
    while w <= len(returns) // 2:
        window_sizes.append(w)
        w = int(w * 1.5)  # Non-uniform spacing for better fit

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
            r = deviations.max() - deviations.min()  # Range
            s = segment.std(ddof=1)                    # Std dev
            if s > 0:
                rs_list.append(r / s)

        if rs_list:
            rs_values.append((w, np.mean(rs_list)))

    if len(rs_values) < 3:
        return 0.5

    # Log-log regression: log(R/S) = H * log(n) + c
    log_n = np.log([v[0] for v in rs_values])
    log_rs = np.log([v[1] for v in rs_values])

    # Simple linear regression
    n_pts = len(log_n)
    sum_x = log_n.sum()
    sum_y = log_rs.sum()
    sum_xy = (log_n * log_rs).sum()
    sum_x2 = (log_n ** 2).sum()

    denom = n_pts * sum_x2 - sum_x ** 2
    if abs(denom) < 1e-12:
        return 0.5

    H = (n_pts * sum_xy - sum_x * sum_y) / denom

    # Clamp to valid range
    return float(np.clip(H, 0.01, 0.99))

def compute_features(records, idx):
    """Compute all features at index idx (backward-looking only)."""
    if idx < SEQUENCE_LENGTH:
        return None

    rec = records[idx]
    close = rec["close"]

    # Price history for lookback calculations
    closes = [records[i]["close"] for i in range(max(0, idx - 200), idx + 1)]
    volumes = [records[i]["volume"] for i in range(max(0, idx - 200), idx + 1)]

    features = {}

    # -- TECHNICAL (same as crypto_predictor) --
    for w in [7, 14, 30, 60, 90, 200]:
        if len(closes) > w:
            features[f"return_{w}d"] = (closes[-1] - closes[-1 - w]) / closes[-1 - w] * 100

    # Moving averages
    for w in [7, 14, 30, 50, 200]:
        if len(closes) >= w:
            ma = sum(closes[-w:]) / w
            features[f"price_vs_ma_{w}d"] = (close - ma) / ma * 100

    # Volatility
    if len(closes) >= 31:
        daily_rets = [(closes[-i] - closes[-i-1]) / closes[-i-1] for i in range(1, 31)]
        features["volatility_30d"] = float(np.std(daily_rets)) * math.sqrt(365) * 100

    if len(closes) >= 8:
        daily_rets_7 = [(closes[-i] - closes[-i-1]) / closes[-i-1] for i in range(1, 8)]
        features["volatility_7d"] = float(np.std(daily_rets_7)) * math.sqrt(365) * 100

    # RSI 14
    if len(closes) >= 15:
        gains = [max(closes[-i] - closes[-i-1], 0) for i in range(1, 15)]
        losses = [max(closes[-i-1] - closes[-i], 0) for i in range(1, 15)]
        avg_gain = sum(gains) / 14
        avg_loss = sum(losses) / 14
        if avg_loss > 0:
            features["rsi_14"] = 100 - (100 / (1 + avg_gain / avg_loss))
        else:
            features["rsi_14"] = 100

    # Bollinger Band position
    if len(closes) >= 20:
        bb_ma = sum(closes[-20:]) / 20
        bb_std = float(np.std(closes[-20:]))
        if bb_std > 0:
            features["bb_position"] = (close - (bb_ma - 2 * bb_std)) / (4 * bb_std)
            features["bb_width"] = 4 * bb_std / bb_ma * 100

    # Volume ratio
    if len(volumes) >= 30:
        avg_vol = sum(volumes[-30:]) / 30
        features["volume_ratio"] = volumes[-1] / avg_vol if avg_vol > 0 else 1

    # ATH distance
    all_highs = [records[i]["high"] for i in range(idx + 1)]
    features["ath_distance_pct"] = (close - max(all_highs)) / max(all_highs) * 100

    # Drawdown from 30d high
    if len(closes) >= 30:
        features["drawdown_30d"] = (close - max(closes[-30:])) / max(closes[-30:]) * 100

    # -- QUANTAMENTAL --
    features["fear_greed"] = rec["fear_greed"]
    features["fear_greed_z"] = (rec["fear_greed"] - 50) / 25  # Normalized

    features["funding_rate"] = rec["funding_avg"] * 10000  # In basis points
    features["funding_cum_7d"] = sum(
        records[i].get("funding_avg", 0) for i in range(max(0, idx - 7), idx + 1)
    ) * 10000

    # Fear & Greed momentum
    if idx >= 7:
        fg_7_ago = records[idx - 7].get("fear_greed", 50)
        features["fg_momentum_7d"] = rec["fear_greed"] - fg_7_ago

    # On-chain features (BTC)
    if rec.get("hash_rate", 0) > 0:
        features["hash_rate"] = rec["hash_rate"]
        features["active_addresses"] = rec.get("active_addresses", 0)
        features["tx_count"] = rec.get("tx_count", 0)

        # Hash rate momentum
        if idx >= 30 and records[idx - 30].get("hash_rate", 0) > 0:
            features["hash_rate_change_30d"] = (
                (rec["hash_rate"] - records[idx - 30]["hash_rate"])
                / records[idx - 30]["hash_rate"] * 100
            )

        # Active address momentum
        if idx >= 7 and records[idx - 7].get("active_addresses", 0) > 0:
            features["addr_change_7d"] = (
                (rec.get("active_addresses", 0) - records[idx - 7].get("active_addresses", 0))
                / max(records[idx - 7].get("active_addresses", 1), 1) * 100
            )

    # Volume-price divergence
    if len(closes) >= 14 and len(volumes) >= 14:
        price_change = (closes[-1] - closes[-14]) / closes[-14]
        vol_change = (sum(volumes[-7:]) / 7) / (sum(volumes[-14:-7]) / 7) - 1
        features["vol_price_divergence"] = vol_change - price_change

    # -- VOLTERRA MEMORY: Multi-timescale Hurst exponent --
    # H controls the kernel decay K(t,s) ~ (t-s)^(H-1/2)
    # H > 0.5 = trending (momentum), H < 0.5 = mean-reverting
    # Different timescales reveal regime transitions

    # Short-term memory (30 bars) -- captures recent regime
    if len(closes) >= 30:
        features["hurst_30d"] = estimate_hurst(closes[-30:], min_window=5)

    # Medium-term memory (60 bars) -- intermediate regime
    if len(closes) >= 60:
        features["hurst_60d"] = estimate_hurst(closes[-60:], min_window=8)

    # Long-term memory (120 bars) -- structural regime
    if len(closes) >= 120:
        features["hurst_120d"] = estimate_hurst(closes[-120:], min_window=10)

    # Full history memory (200 bars) -- macro regime
    if len(closes) >= 200:
        features["hurst_200d"] = estimate_hurst(closes[-200:], min_window=10)

    # Regime transition signals: divergence between timescales
    if "hurst_30d" in features and "hurst_120d" in features:
        # Short H trending but long H mean-reverting = potential reversal
        # Short H mean-reverting but long H trending = potential breakout
        features["hurst_divergence"] = features["hurst_30d"] - features["hurst_120d"]

    if "hurst_30d" in features and "hurst_200d" in features:
        features["hurst_divergence_wide"] = features["hurst_30d"] - features["hurst_200d"]

    # Hurst regime classification (for the quantamental model)
    if "hurst_60d" in features:
        h = features["hurst_60d"]
        # Distance from random walk (0.5) -- how strong the regime is
        features["hurst_regime_strength"] = abs(h - 0.5) * 2  # 0=random, 1=strong regime
        # Directional: positive=trending, negative=mean-reverting
        features["hurst_regime_direction"] = (h - 0.5) * 2

    # Volume-weighted Hurst: compute Hurst on volume series too
    if len(volumes) >= 60:
        features["hurst_volume_60d"] = estimate_hurst(volumes[-60:], min_window=8)

        # Volume memory vs price memory divergence
        if "hurst_60d" in features:
            features["hurst_vol_price_divergence"] = (
                features["hurst_volume_60d"] - features["hurst_60d"]
            )

    return features


def build_lstm_sequences(records, feature_cols, horizon):
    """Build LSTM input sequences and labels.

    Each sequence is SEQUENCE_LENGTH days of normalized price/volume data.
    Label is whether price went up over the horizon.
    """
    sequences = []
    labels = []
    dates = []

    # Precompute rolling Hurst exponent for each record
    # This gives the LSTM explicit regime awareness
    closes_all = [r["close"] for r in records]
    rolling_hurst = []
    for i in range(len(records)):
        if i >= 30:
            h = estimate_hurst(closes_all[i-30:i+1], min_window=5)
            rolling_hurst.append(h)
        else:
            rolling_hurst.append(0.5)  # Default to random walk

    # Columns for LSTM: normalized OHLCV + Fear & Greed + funding + Hurst
    seq_cols = ["close", "high", "low", "volume", "fear_greed", "funding_avg"]

    for i in range(SEQUENCE_LENGTH, len(records) - horizon):
        # Build sequence
        seq = []
        for j in range(i - SEQUENCE_LENGTH, i):
            row = []
            for col in seq_cols:
                val = records[j].get(col, 0)
                row.append(float(val) if val else 0)
            # Add rolling Hurst as 7th feature
            row.append(rolling_hurst[j])
            seq.append(row)

        # Normalize sequence: divide by the last close price
        last_close = records[i - 1]["close"]
        seq_arr = np.array(seq, dtype=np.float32)
        # Normalize price columns by last close
        for ci in range(3):  # close, high, low
            seq_arr[:, ci] = seq_arr[:, ci] / last_close - 1.0
        # Normalize volume by mean
        vol_mean = seq_arr[:, 3].mean()
        if vol_mean > 0:
            seq_arr[:, 3] = seq_arr[:, 3] / vol_mean - 1.0
        # Fear & Greed: 0-100 -> -1 to 1
        seq_arr[:, 4] = seq_arr[:, 4] / 50 - 1.0
        # Funding rate: already small, just scale
        seq_arr[:, 5] = seq_arr[:, 5] * 10000
        # Hurst: center around 0.5 and scale
        seq_arr[:, 6] = (seq_arr[:, 6] - 0.5) * 4  # Range ~[-2, 2]

        sequences.append(seq_arr)

        # Label
        future_close = records[i + horizon - 1]["close"]
        current_close = records[i - 1]["close"]
        labels.append(1 if future_close > current_close else 0)
        dates.append(records[i - 1]["date"])

    return np.array(sequences), np.array(labels), dates


# ===================================================================
# MULTI-HORIZON QUANTILE PREDICTION
# ===================================================================

QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]  # Prediction quantiles
MULTI_HORIZONS = [1, 3, 7, 14, 30]           # All horizons predicted simultaneously


def build_multi_horizon_sequences(records):
    """Build sequences with multi-horizon return labels for all horizons at once.

    Returns:
        sequences: (N, SEQUENCE_LENGTH, n_features) input arrays
        returns: (N, len(MULTI_HORIZONS)) actual future returns per horizon
        dates: list of date strings
    """
    sequences = []
    returns = []
    dates = []

    max_horizon = max(MULTI_HORIZONS)

    # Precompute rolling Hurst
    closes_all = [r["close"] for r in records]
    rolling_hurst = []
    for i in range(len(records)):
        if i >= 30:
            h = estimate_hurst(closes_all[i-30:i+1], min_window=5)
            rolling_hurst.append(h)
        else:
            rolling_hurst.append(0.5)

    seq_cols = ["close", "high", "low", "volume", "fear_greed", "funding_avg"]

    for i in range(SEQUENCE_LENGTH, len(records) - max_horizon):
        # Build sequence (same as binary version)
        seq = []
        for j in range(i - SEQUENCE_LENGTH, i):
            row = []
            for col in seq_cols:
                val = records[j].get(col, 0)
                row.append(float(val) if val else 0)
            row.append(rolling_hurst[j])
            seq.append(row)

        last_close = records[i - 1]["close"]
        seq_arr = np.array(seq, dtype=np.float32)

        # Normalize
        for ci in range(3):
            seq_arr[:, ci] = seq_arr[:, ci] / last_close - 1.0
        vol_mean = seq_arr[:, 3].mean()
        if vol_mean > 0:
            seq_arr[:, 3] = seq_arr[:, 3] / vol_mean - 1.0
        seq_arr[:, 4] = seq_arr[:, 4] / 50 - 1.0
        seq_arr[:, 5] = seq_arr[:, 5] * 10000
        seq_arr[:, 6] = (seq_arr[:, 6] - 0.5) * 4

        sequences.append(seq_arr)

        # Multi-horizon returns (percentage)
        current_close = records[i - 1]["close"]
        horizon_returns = []
        for h in MULTI_HORIZONS:
            future_close = records[i + h - 1]["close"]
            ret = (future_close - current_close) / current_close * 100
            horizon_returns.append(ret)
        returns.append(horizon_returns)
        dates.append(records[i - 1]["date"])

    return np.array(sequences), np.array(returns, dtype=np.float32), dates


def train_multi_horizon_lstm(sequences, returns, asset):
    """Train multi-horizon quantile LSTM.

    Outputs: for each horizon, predict p10/p25/p50/p75/p90 of return distribution.
    Uses pinball (quantile) loss instead of BCE.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print(f"    PyTorch not installed. pip install torch")
        return None

    n_horizons = len(MULTI_HORIZONS)
    n_quantiles = len(QUANTILES)

    # Split
    split = int(len(sequences) * TRAIN_SPLIT)
    X_train = torch.FloatTensor(sequences[:split])
    y_train = torch.FloatTensor(returns[:split])
    X_test = torch.FloatTensor(sequences[split:])
    y_test = torch.FloatTensor(returns[split:])

    print(f"    Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"    Horizons: {MULTI_HORIZONS} | Quantiles: {QUANTILES}")
    print(f"    Output shape: {n_horizons} horizons x {n_quantiles} quantiles "
          f"= {n_horizons * n_quantiles} outputs")

    # Model: shared LSTM backbone + per-horizon quantile heads
    class MultiHorizonQuantileLSTM(nn.Module):
        def __init__(self, input_size, hidden_size=96, num_layers=2,
                     dropout=0.3, n_horizons=5, n_quantiles=5):
            super().__init__()
            self.n_horizons = n_horizons
            self.n_quantiles = n_quantiles

            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, dropout=dropout)

            # Shared intermediate layer
            self.shared = nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
            )

            # Per-horizon heads: each outputs n_quantiles values
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

            # Each head outputs quantiles for one horizon
            outputs = []
            for head in self.heads:
                outputs.append(head(shared))

            # Shape: (batch, n_horizons, n_quantiles)
            return torch.stack(outputs, dim=1)

    # Pinball (quantile) loss
    def pinball_loss(predictions, targets, quantiles_tensor):
        """Quantile regression loss.
        predictions: (batch, n_horizons, n_quantiles)
        targets: (batch, n_horizons)
        quantiles_tensor: (n_quantiles,)
        """
        targets_expanded = targets.unsqueeze(-1)  # (batch, n_horizons, 1)
        errors = targets_expanded - predictions   # (batch, n_horizons, n_quantiles)

        # Asymmetric loss: overestimate penalized by (1-q), underestimate by q
        loss = torch.max(
            quantiles_tensor * errors,
            (quantiles_tensor - 1) * errors
        )
        return loss.mean()

    input_size = X_train.shape[2]
    model = MultiHorizonQuantileLSTM(input_size, n_horizons=n_horizons,
                                      n_quantiles=n_quantiles)
    quantiles_tensor = torch.FloatTensor(QUANTILES)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8)

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    best_test_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    epochs = 150

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = pinball_loss(pred, batch_y, quantiles_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # Evaluate
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test)
            test_loss = pinball_loss(test_pred, y_test, quantiles_tensor).item()

            # Also compute directional accuracy per horizon (using p50 median)
            median_idx = QUANTILES.index(0.50)
            median_preds = test_pred[:, :, median_idx]  # (batch, n_horizons)
            direction_correct = ((median_preds > 0) == (y_test > 0)).float().mean(dim=0)

        scheduler.step(test_loss)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 25 == 0 or epoch == 0:
            dir_strs = " ".join(f"{MULTI_HORIZONS[h]}d={direction_correct[h]:.0%}"
                                for h in range(n_horizons))
            print(f"    Epoch {epoch+1:>3d}: loss={total_loss/len(loader):.4f} "
                  f"test_loss={test_loss:.4f} dir=[{dir_strs}]")

        if patience_counter >= 20:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    # Load best and final eval
    if best_model_state:
        model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        final_pred = model(X_test)
        median_idx = QUANTILES.index(0.50)

        print(f"\n    Final test results:")
        for hi, h in enumerate(MULTI_HORIZONS):
            median_preds = final_pred[:, hi, median_idx]
            actual = y_test[:, hi]
            dir_acc = ((median_preds > 0) == (actual > 0)).float().mean()

            # Calibration: what fraction of actuals fall within p10-p90?
            p10 = final_pred[:, hi, 0]
            p90 = final_pred[:, hi, -1]
            coverage = ((actual >= p10) & (actual <= p90)).float().mean()

            # Mean absolute error of median prediction
            mae = (median_preds - actual).abs().mean()

            print(f"      {h:>2d}d: direction={dir_acc:.1%} | "
                  f"p10-p90 coverage={coverage:.1%} (ideal=80%) | "
                  f"MAE={mae:.2f}%")

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


def train_quantamental_regression(feature_rows, asset, horizons=None):
    """Train XGBoost quantile regression for each horizon.

    Instead of classification (up/down), predicts return quantiles.
    """
    if horizons is None:
        horizons = MULTI_HORIZONS

    results = {}

    for horizon in horizons:
        target_col = f"target_{horizon}d_return"
        valid = [f for f in feature_rows if target_col in f]

        if len(valid) < 100:
            print(f"    {horizon}d: Only {len(valid)} rows, need 100+")
            continue

        feat_cols = sorted([k for k in valid[0] if not k.startswith("target_")
                            and k not in ("date", "close")
                            and isinstance(valid[0][k], (int, float))])

        X = np.array([[f.get(c, 0) for c in feat_cols] for f in valid])
        y = np.array([f[target_col] for f in valid])
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        split = int(len(X) * TRAIN_SPLIT)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        print(f"\n    {horizon}d Quantile Regression: train={len(X_train)} test={len(X_test)}")

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

            # Directional accuracy from median
            median_pred = test_preds[0.50]
            dir_acc = ((median_pred > 0) == (y_test > 0)).mean()

            # Coverage
            coverage = ((y_test >= test_preds[0.10]) &
                        (y_test <= test_preds[0.90])).mean()

            mae = np.abs(median_pred - y_test).mean()

            print(f"      direction={dir_acc:.1%} | coverage={coverage:.1%} | MAE={mae:.2f}%")

            # Top features from median model
            importances = quantile_models[0.50].feature_importances_
            top = sorted(zip(feat_cols, importances), key=lambda x: -x[1])[:5]
            print(f"      Top: {', '.join(f'{n}={v:.3f}' for n, v in top)}")

            results[horizon] = {
                "models": quantile_models,
                "feature_cols": feat_cols,
                "dir_accuracy": dir_acc,
                "coverage": coverage,
                "mae": mae,
            }

        except Exception as e:
            print(f"      Error: {e}")

    return results


def cmd_build_features(args):
    """Build features for both models."""
    asset = args.asset.upper()

    print(f"\n{'='*70}")
    print(f"  FEATURE ENGINEERING -- {asset}")
    print(f"{'='*70}")

    records = load_all_data(asset)
    if not records:
        return

    # Build tabular features for quantamental model
    feature_rows = []
    for i in range(SEQUENCE_LENGTH, len(records)):
        feats = compute_features(records, i)
        if feats:
            feats["date"] = records[i]["date"]
            feats["close"] = records[i]["close"]

            # Add labels for each horizon (union of binary and multi-horizon sets)
            all_horizons = sorted(set(PREDICTION_HORIZONS + MULTI_HORIZONS))
            for h in all_horizons:
                if i + h < len(records):
                    future = records[i + h]["close"]
                    feats[f"target_{h}d_return"] = (future - records[i]["close"]) / records[i]["close"] * 100
                    feats[f"target_{h}d_up"] = 1 if future > records[i]["close"] else 0

            feature_rows.append(feats)

    print(f"\n  Tabular features: {len(feature_rows)} rows")
    if feature_rows:
        feat_cols = [k for k in feature_rows[0] if not k.startswith("target_")
                     and k not in ("date", "close")]
        print(f"  Feature columns: {len(feat_cols)}")
        print(f"  Columns: {', '.join(sorted(feat_cols)[:15])}...")

    # Build LSTM sequences
    for h in PREDICTION_HORIZONS:
        seqs, labels, dates = build_lstm_sequences(records, None, h)
        if len(seqs) > 0:
            print(f"  LSTM {h}d: {len(seqs)} sequences, shape={seqs[0].shape}, "
                  f"up_rate={labels.mean():.1%}")

    # Save features
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump({
        "feature_rows": feature_rows,
        "records": records,
        "asset": asset,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }, MODEL_DIR / f"{asset}_features.joblib")

    print(f"  Saved to {MODEL_DIR / f'{asset}_features.joblib'}")
    print(f"{'='*70}")


# ===================================================================
# MODEL TRAINING
# ===================================================================

def train_lstm(sequences, labels, asset, horizon):
    """Train LSTM model using PyTorch."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print(f"    PyTorch not installed. pip install torch")
        return None

    # Split
    split = int(len(sequences) * TRAIN_SPLIT)
    X_train = torch.FloatTensor(sequences[:split])
    y_train = torch.FloatTensor(labels[:split])
    X_test = torch.FloatTensor(sequences[split:])
    y_test = torch.FloatTensor(labels[split:])

    print(f"    Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"    Sequence shape: {X_train.shape}")

    # Model
    class CryptoLSTM(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, dropout=dropout)
            self.fc1 = nn.Linear(hidden_size, 32)
            self.fc2 = nn.Linear(32, 1)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.3)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            last_hidden = lstm_out[:, -1, :]
            x = self.dropout(self.relu(self.fc1(last_hidden)))
            return self.sigmoid(self.fc2(x)).squeeze()

    input_size = X_train.shape[2]
    model = CryptoLSTM(input_size)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    # Training
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    best_test_acc = 0
    best_model_state = None
    epochs = 100
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # Evaluate
        model.eval()
        with torch.no_grad():
            train_pred = (model(X_train) > 0.5).float()
            train_acc = (train_pred == y_train).float().mean()
            test_pred = (model(X_test) > 0.5).float()
            test_acc = (test_pred == y_test).float().mean()

        scheduler.step(total_loss)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:>3d}: loss={total_loss/len(loader):.4f} "
                  f"train={train_acc:.1%} test={test_acc:.1%}")

        if patience_counter >= 15:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        final_test_pred = model(X_test)
        final_test_binary = (final_test_pred > 0.5).float()
        final_acc = (final_test_binary == y_test).float().mean()

    print(f"    Best test accuracy: {best_test_acc:.1%}")

    return {
        "model_state": best_model_state,
        "input_size": input_size,
        "test_accuracy": float(best_test_acc),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }


def train_quantamental(feature_rows, asset, horizon):
    """Train XGBoost quantamental model."""
    target_col = f"target_{horizon}d_up"

    valid = [f for f in feature_rows if target_col in f]
    if len(valid) < 100:
        print(f"    Only {len(valid)} labeled rows, need 100+")
        return None

    feat_cols = sorted([k for k in valid[0] if not k.startswith("target_")
                        and k not in ("date", "close")
                        and isinstance(valid[0][k], (int, float))])

    X = np.array([[f.get(c, 0) for c in feat_cols] for f in valid])
    y = np.array([f[target_col] for f in valid])
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    split = int(len(X) * TRAIN_SPLIT)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"    Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"    Features: {len(feat_cols)}")
    print(f"    Base rate: {y_train.mean():.1%}")

    try:
        import xgboost as xgb

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)

        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)

        # Top features
        importances = model.feature_importances_
        top = sorted(zip(feat_cols, importances), key=lambda x: -x[1])[:5]

        print(f"    Train: {train_acc:.1%} | Test: {test_acc:.1%}")
        print(f"    Top features: {', '.join(f'{n}={v:.3f}' for n, v in top)}")

        return {
            "model": model,
            "feature_cols": feat_cols,
            "test_accuracy": test_acc,
            "top_features": top,
        }

    except ImportError:
        print(f"    XGBoost not installed")
        return None


def cmd_train(args):
    """Train both LSTM and quantamental models — multi-horizon quantile."""
    asset = args.asset.upper()
    mode = getattr(args, "mode", "quantile")  # "quantile" (new) or "binary" (legacy)

    print(f"\n{'='*70}")
    print(f"  MULTI-HORIZON QUANTILE TRAINING -- {asset}")
    print(f"  Horizons: {MULTI_HORIZONS}")
    print(f"  Quantiles: {QUANTILES}")
    print(f"{'='*70}")

    import joblib
    feat_path = MODEL_DIR / f"{asset}_features.joblib"
    if not feat_path.exists():
        print(f"  Run build-features first")
        return

    data = joblib.load(feat_path)
    records = data["records"]
    feature_rows = data["feature_rows"]

    # ── MULTI-HORIZON LSTM ──
    print(f"\n  {'='*60}")
    print(f"  MULTI-HORIZON QUANTILE LSTM")
    print(f"  {'='*60}")

    seqs, returns, dates = build_multi_horizon_sequences(records)
    print(f"  Sequences: {len(seqs)}, shape={seqs[0].shape}")
    print(f"  Return targets shape: {returns.shape}")

    lstm_mh_result = None
    if len(seqs) > 100:
        lstm_mh_result = train_multi_horizon_lstm(seqs, returns, asset)

        if lstm_mh_result and lstm_mh_result.get("model_state"):
            try:
                import torch
                torch.save(lstm_mh_result["model_state"],
                           MODEL_DIR / f"{asset}_multi_horizon_lstm.pt")
                print(f"\n  Multi-horizon LSTM saved")
            except Exception as e:
                print(f"  Save error: {e}")
    else:
        print(f"  Not enough sequences ({len(seqs)})")

    # ── MULTI-HORIZON QUANTAMENTAL ──
    print(f"\n  {'='*60}")
    print(f"  MULTI-HORIZON QUANTILE XGBOOST")
    print(f"  {'='*60}")

    quant_mh_result = train_quantamental_regression(feature_rows, asset)

    # Save everything
    save_data = {
        "asset": asset,
        "mode": "multi_horizon_quantile",
        "horizons": MULTI_HORIZONS,
        "quantiles": QUANTILES,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "lstm_multi_horizon": lstm_mh_result if lstm_mh_result else None,
        "quantamental_multi_horizon": {
            h: {k: v for k, v in r.items() if k != "models"}
            for h, r in quant_mh_result.items()
        } if quant_mh_result else None,
    }

    # Save quantamental models separately (they contain sklearn objects)
    if quant_mh_result:
        joblib.dump(quant_mh_result,
                    MODEL_DIR / f"{asset}_quant_mh_models.joblib")

    joblib.dump(save_data, MODEL_DIR / f"{asset}_multi_horizon.joblib")
    print(f"\n  Saved: {MODEL_DIR / f'{asset}_multi_horizon.joblib'}")

    # Also train legacy binary models for backward compatibility
    print(f"\n  {'='*60}")
    print(f"  LEGACY BINARY MODELS (backward compatible)")
    print(f"  {'='*60}")

    for horizon in PREDICTION_HORIZONS:
        print(f"\n  -- {horizon}d binary --")
        seqs_bin, labels, _ = build_lstm_sequences(records, None, horizon)
        if len(seqs_bin) > 100:
            lstm_result = train_lstm(seqs_bin, labels, asset, horizon)
        else:
            lstm_result = None

        quant_result = train_quantamental(feature_rows, asset, horizon)

        save_bin = {
            "asset": asset, "horizon": horizon,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "quantamental": quant_result,
        }
        if lstm_result:
            save_bin["lstm"] = {
                "input_size": lstm_result["input_size"],
                "test_accuracy": lstm_result["test_accuracy"],
            }
            try:
                import torch
                torch.save(lstm_result["model_state"],
                           MODEL_DIR / f"{asset}_{horizon}d_lstm.pt")
            except Exception:
                pass

        joblib.dump(save_bin, MODEL_DIR / f"{asset}_{horizon}d_ensemble.joblib")

    print(f"\n{'='*70}")


# ===================================================================
# PREDICTION
# ===================================================================

def cmd_predict(args):
    """Generate predictions using the ensemble."""
    asset = args.asset.upper()

    print(f"\n{'='*70}")
    print(f"  ENSEMBLE PREDICTION -- {asset}")
    print(f"{'='*70}")

    import joblib

    records = load_all_data(asset)
    if not records:
        return

    current_price = records[-1]["close"]
    print(f"  Current price: ${current_price:,.2f}")
    print(f"  Date: {records[-1]['date']}")

    for horizon in PREDICTION_HORIZONS:
        model_path = MODEL_DIR / f"{asset}_{horizon}d_ensemble.joblib"
        if not model_path.exists():
            print(f"\n  {horizon}d: No model. Run train first.")
            continue

        saved = joblib.load(model_path)

        print(f"\n  -- {horizon}-Day Prediction --")

        # Quantamental prediction
        quant_prob = 0.5
        quant_conf = 0
        if saved.get("quantamental") and saved["quantamental"].get("model"):
            model = saved["quantamental"]["model"]
            feat_cols = saved["quantamental"]["feature_cols"]

            feats = compute_features(records, len(records) - 1)
            if feats:
                X = np.array([[feats.get(c, 0) for c in feat_cols]])
                X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
                quant_prob = float(model.predict_proba(X)[0][1])
                quant_conf = abs(quant_prob - 0.5) * 2
                print(f"    Quantamental: {quant_prob:.1%} up (conf={quant_conf:.1%})")

        # LSTM prediction
        lstm_prob = 0.5
        lstm_conf = 0
        lstm_path = MODEL_DIR / f"{asset}_{horizon}d_lstm.pt"
        if lstm_path.exists() and saved.get("lstm"):
            try:
                import torch
                import torch.nn as nn

                class CryptoLSTM(nn.Module):
                    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
                        super().__init__()
                        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                            batch_first=True, dropout=dropout)
                        self.fc1 = nn.Linear(hidden_size, 32)
                        self.fc2 = nn.Linear(32, 1)
                        self.relu = nn.ReLU()
                        self.dropout = nn.Dropout(0.3)
                        self.sigmoid = nn.Sigmoid()

                    def forward(self, x):
                        lstm_out, _ = self.lstm(x)
                        last_hidden = lstm_out[:, -1, :]
                        x = self.dropout(self.relu(self.fc1(last_hidden)))
                        return self.sigmoid(self.fc2(x)).squeeze()

                input_size = saved["lstm"]["input_size"]
                model = CryptoLSTM(input_size)
                model.load_state_dict(torch.load(lstm_path, weights_only=True))
                model.eval()

                # Build current sequence
                seqs, _, _ = build_lstm_sequences(records, None, horizon)
                if len(seqs) > 0:
                    with torch.no_grad():
                        X = torch.FloatTensor(seqs[-1:])
                        lstm_prob = float(model(X))
                        lstm_conf = abs(lstm_prob - 0.5) * 2
                        print(f"    LSTM:          {lstm_prob:.1%} up (conf={lstm_conf:.1%})")

            except Exception as e:
                print(f"    LSTM error: {e}")

        # Ensemble
        if quant_conf > 0 and lstm_conf > 0:
            # Weighted by confidence
            total_conf = quant_conf + lstm_conf
            ensemble_prob = (quant_prob * quant_conf + lstm_prob * lstm_conf) / total_conf
            agreement = 1 - abs(quant_prob - lstm_prob)
        elif quant_conf > 0:
            ensemble_prob = quant_prob
            agreement = 0.5
        elif lstm_conf > 0:
            ensemble_prob = lstm_prob
            agreement = 0.5
        else:
            ensemble_prob = 0.5
            agreement = 0

        direction = "UP" if ensemble_prob > 0.5 else "DOWN"
        signal_strength = "STRONG" if agreement > 0.7 and abs(ensemble_prob - 0.5) > 0.15 else \
                          "MODERATE" if abs(ensemble_prob - 0.5) > 0.1 else "WEAK"

        print(f"    Ensemble:      {ensemble_prob:.1%} up")
        print(f"    Direction:     {'UP' if direction == 'UP' else 'DOWN'} {direction}")
        print(f"    Agreement:     {agreement:.1%}")
        print(f"    Signal:        {signal_strength}")

    print(f"\n{'='*70}")


def cmd_markets(args):
    """Compare predictions vs Polymarket crypto markets."""
    print(f"\n{'='*70}")
    print(f"  ENSEMBLE vs POLYMARKET CRYPTO MARKETS")
    print(f"{'='*70}")

    import joblib

    # Get predictions for each asset
    for asset in SUPPORTED_ASSETS:
        records = load_all_data(asset)
        if not records:
            continue

        current_price = records[-1]["close"]
        print(f"\n  {asset} @ ${current_price:,.2f}")

        for horizon in PREDICTION_HORIZONS:
            model_path = MODEL_DIR / f"{asset}_{horizon}d_ensemble.joblib"
            if not model_path.exists():
                continue

            saved = joblib.load(model_path)
            if saved.get("quantamental") and saved["quantamental"].get("model"):
                model = saved["quantamental"]["model"]
                feat_cols = saved["quantamental"]["feature_cols"]
                feats = compute_features(records, len(records) - 1)
                if feats:
                    X = np.array([[feats.get(c, 0) for c in feat_cols]])
                    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
                    prob = float(model.predict_proba(X)[0][1])
                    print(f"    {horizon}d model: {prob:.1%} probability of going up")

    # Fetch crypto Polymarket markets
    print(f"\n  Polymarket crypto markets:")
    try:
        r = requests.get(f"{GAMMA_API}/markets", params={
            "closed": "false", "active": "true", "limit": 200,
        }, timeout=15)
        markets = r.json()

        crypto_kw = ["bitcoin", "btc", "ethereum", "eth", "crypto", "price"]
        for m in markets:
            q = (m.get("question", "") or "").lower()
            if any(kw in q for kw in crypto_kw):
                token_ids = json.loads(m.get("clobTokenIds", "[]"))
                if token_ids:
                    try:
                        r2 = requests.get(f"{CLOB_API}/midpoint",
                                          params={"token_id": token_ids[0]}, timeout=5)
                        d = r2.json()
                        mid = float(d.get("mid", 0.5) if isinstance(d, dict) else d)
                    except Exception:
                        mid = 0.5

                    print(f"    {m.get('question', '')[:55]:<56s} {mid:.0%}")
                    time.sleep(0.05)

    except Exception as e:
        print(f"    Error fetching markets: {e}")

    print(f"\n{'='*70}")


def cmd_backtest(args):
    """Walk-forward backtest of the ensemble."""
    asset = args.asset.upper()

    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD BACKTEST -- {asset}")
    print(f"{'='*70}")

    records = load_all_data(asset)
    if not records:
        return

    for horizon in PREDICTION_HORIZONS:
        print(f"\n  -- {horizon}-Day Horizon --")

        # Use the quantamental model for backtesting
        feat_path = MODEL_DIR / f"{asset}_features.joblib"
        if not feat_path.exists():
            print(f"    No features. Run build-features first.")
            continue

        import joblib
        data = joblib.load(feat_path)
        feature_rows = data["feature_rows"]

        target_col = f"target_{horizon}d_up"
        valid = [f for f in feature_rows if target_col in f]

        if len(valid) < 200:
            print(f"    Need 200+ rows, have {len(valid)}")
            continue

        feat_cols = sorted([k for k in valid[0] if not k.startswith("target_")
                            and k not in ("date", "close")
                            and isinstance(valid[0][k], (int, float))])

        X = np.array([[f.get(c, 0) for c in feat_cols] for f in valid])
        y = np.array([f[target_col] for f in valid])
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        # Walk-forward: train on expanding window, predict next batch
        window_size = 100
        step_size = 20
        results = []

        for start in range(window_size, len(X) - step_size, step_size):
            X_train = X[:start]
            y_train = y[:start]
            X_test = X[start:start + step_size]
            y_test = y[start:start + step_size]

            try:
                import xgboost as xgb
                model = xgb.XGBClassifier(
                    n_estimators=100, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.7,
                    random_state=42, eval_metric="logloss",
                )
                model.fit(X_train, y_train, verbose=False)
                preds = model.predict(X_test)
                acc = (preds == y_test).mean()
                results.append({
                    "window_end": start,
                    "accuracy": acc,
                    "n_test": len(y_test),
                    "base_rate": y_test.mean(),
                })
            except Exception:
                pass

        if results:
            avg_acc = np.mean([r["accuracy"] for r in results])
            avg_base = np.mean([r["base_rate"] for r in results])
            edge = avg_acc - avg_base

            print(f"    Walk-forward windows: {len(results)}")
            print(f"    Average accuracy: {avg_acc:.1%}")
            print(f"    Average base rate: {avg_base:.1%}")
            print(f"    Edge over base: {edge:+.1%}")

            if edge > 0.05:
                print(f"    Consistent edge detected!")
            elif edge > 0:
                print(f"    Marginal edge -- needs more data")
            else:
                print(f"    No edge -- model not beating base rate")

    print(f"\n{'='*70}")


def cmd_confluence(args):
    """Show mean-reversion confluence signals using multi-horizon predictions.

    The dual-horizon filter for mean-reversion timing:
    1. Z-score threshold hit (spread is extended)
    2. Near-term LSTM predicts direction aligning with reversion start
    3. Long-term LSTM predicts the reversion hasn't fully played out

    For LONG (z-score very negative, asset is cheap):
      - Short-term (1-7d) predicts UP = bounce is starting
      - Long-term (14-30d) predicts continued UP = room to capture

    For SHORT (z-score very positive, asset is expensive):
      - Short-term (1-7d) predicts DOWN = top is rolling over
      - Long-term (14-30d) predicts continued DOWN = reversion underway
    """
    asset = args.asset.upper()
    zscore_threshold = getattr(args, "zscore", 2.0)
    short_horizon_idx = 0   # 1d from MULTI_HORIZONS
    long_horizon_idx = -1   # 30d from MULTI_HORIZONS

    print(f"\n{'='*70}")
    print(f"  CONFLUENCE SIGNALS -- {asset}")
    print(f"  Z-score threshold: +/-{zscore_threshold}")
    print(f"  Short horizon: {MULTI_HORIZONS[short_horizon_idx]}d")
    print(f"  Long horizon: {MULTI_HORIZONS[long_horizon_idx]}d")
    print(f"{'='*70}")

    import joblib

    records = load_all_data(asset)
    if not records:
        return

    current_price = records[-1]["close"]
    print(f"  Current price: ${current_price:,.2f}")

    # Compute current z-score (price vs 30d moving average)
    closes = [r["close"] for r in records]
    if len(closes) >= 30:
        ma30 = sum(closes[-30:]) / 30
        std30 = float(np.std(closes[-30:]))
        zscore = (current_price - ma30) / std30 if std30 > 0 else 0
        print(f"  Z-score (30d): {zscore:+.2f}")
        print(f"  MA30: ${ma30:,.2f} | Std: ${std30:,.2f}")
    else:
        zscore = 0
        print(f"  Insufficient data for z-score")

    # Compute Hurst
    if len(closes) >= 60:
        h60 = estimate_hurst(closes[-60:], min_window=8)
        print(f"  Hurst (60d): {h60:.3f} "
              f"({'trending' if h60 > 0.55 else 'mean-reverting' if h60 < 0.45 else 'random'})")

    # Load multi-horizon model
    mh_path = MODEL_DIR / f"{asset}_multi_horizon.joblib"
    if not mh_path.exists():
        print(f"\n  No multi-horizon model. Run: train --asset {asset}")
        return

    saved = joblib.load(mh_path)

    # Get LSTM multi-horizon prediction
    print(f"\n  Multi-horizon predictions:")
    print(f"  {'Horizon':<8s} {'p10':>8s} {'p25':>8s} {'p50':>8s} "
          f"{'p75':>8s} {'p90':>8s} {'Direction'}")
    print(f"  {'-'*65}")

    lstm_preds = None
    lstm_path = MODEL_DIR / f"{asset}_multi_horizon_lstm.pt"
    if lstm_path.exists() and saved.get("lstm_multi_horizon"):
        try:
            import torch
            import torch.nn as nn

            class MultiHorizonQuantileLSTM(nn.Module):
                def __init__(self, input_size, hidden_size=96, num_layers=2,
                             dropout=0.3, n_horizons=5, n_quantiles=5):
                    super().__init__()
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                        batch_first=True, dropout=dropout)
                    self.shared = nn.Sequential(
                        nn.Linear(hidden_size, 64), nn.ReLU(), nn.Dropout(0.2))
                    self.heads = nn.ModuleList([
                        nn.Sequential(nn.Linear(64, 32), nn.ReLU(),
                                      nn.Linear(32, n_quantiles))
                        for _ in range(n_horizons)])

                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    shared = self.shared(lstm_out[:, -1, :])
                    return torch.stack([h(shared) for h in self.heads], dim=1)

            info = saved["lstm_multi_horizon"]
            model = MultiHorizonQuantileLSTM(
                info["input_size"],
                n_horizons=info["n_horizons"],
                n_quantiles=info["n_quantiles"])
            model.load_state_dict(torch.load(lstm_path, weights_only=True))
            model.eval()

            seqs, _, _ = build_multi_horizon_sequences(records)
            if len(seqs) > 0:
                with torch.no_grad():
                    pred = model(torch.FloatTensor(seqs[-1:]))
                    lstm_preds = pred[0].numpy()  # (n_horizons, n_quantiles)

                for hi, h in enumerate(MULTI_HORIZONS):
                    p = lstm_preds[hi]
                    direction = "UP" if p[2] > 0 else "DOWN"  # p50 = median
                    print(f"  LSTM {h:>2d}d  {p[0]:>+7.1f}% {p[1]:>+7.1f}% "
                          f"{p[2]:>+7.1f}% {p[3]:>+7.1f}% {p[4]:>+7.1f}%  {direction}")

        except Exception as e:
            print(f"  LSTM prediction error: {e}")

    # Get quantamental predictions
    quant_path = MODEL_DIR / f"{asset}_quant_mh_models.joblib"
    quant_preds = {}
    if quant_path.exists():
        try:
            quant_models = joblib.load(quant_path)
            feats = compute_features(records, len(records) - 1)
            if feats:
                for h, hdata in quant_models.items():
                    feat_cols = hdata["feature_cols"]
                    X = np.array([[feats.get(c, 0) for c in feat_cols]])
                    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

                    preds = {}
                    for q, m in hdata["models"].items():
                        preds[q] = float(m.predict(X)[0])
                    quant_preds[h] = preds

                    p = [preds[q] for q in QUANTILES]
                    direction = "UP" if preds[0.50] > 0 else "DOWN"
                    print(f"  XGB  {h:>2d}d  {p[0]:>+7.1f}% {p[1]:>+7.1f}% "
                          f"{p[2]:>+7.1f}% {p[3]:>+7.1f}% {p[4]:>+7.1f}%  {direction}")

        except Exception as e:
            print(f"  Quantamental error: {e}")

    # ── CONFLUENCE CHECK ──
    print(f"\n  {'='*60}")
    print(f"  CONFLUENCE ANALYSIS")
    print(f"  {'='*60}")

    signals = []

    # Check z-score gate
    if abs(zscore) < zscore_threshold:
        print(f"\n  Z-score {zscore:+.2f} is within +/-{zscore_threshold}")
        print(f"  No mean-reversion signal active. Price is not extended.")
    else:
        direction = "LONG" if zscore < 0 else "SHORT"
        print(f"\n  Z-score {zscore:+.2f} exceeds threshold -> {direction} candidate")

        # Check multi-horizon alignment
        if lstm_preds is not None:
            short_median = lstm_preds[short_horizon_idx][2]  # p50
            long_median = lstm_preds[long_horizon_idx][2]

            short_h = MULTI_HORIZONS[short_horizon_idx]
            long_h = MULTI_HORIZONS[long_horizon_idx]

            print(f"\n  LSTM horizons:")
            print(f"    {short_h}d median: {short_median:+.1f}%")
            print(f"    {long_h}d median: {long_median:+.1f}%")

            if direction == "LONG":
                # Want: short-term UP (bounce starting) + long-term UP (room)
                short_ok = short_median > 0
                long_ok = long_median > 0
                confluence = short_ok and long_ok

                print(f"    Short-term UP: {'YES' if short_ok else 'NO'}")
                print(f"    Long-term UP:  {'YES' if long_ok else 'NO'}")

            else:  # SHORT
                short_ok = short_median < 0
                long_ok = long_median < 0
                confluence = short_ok and long_ok

                print(f"    Short-term DOWN: {'YES' if short_ok else 'NO'}")
                print(f"    Long-term DOWN:  {'YES' if long_ok else 'NO'}")

            # Hurst confirmation
            hurst_ok = h60 < 0.45 if len(closes) >= 60 else False

            if confluence and hurst_ok:
                print(f"\n    >>> TRIPLE CONFLUENCE: Z-score + Dual-horizon + Hurst <<<")
                print(f"    >>> STRONG {direction} SIGNAL")
                signals.append({"type": direction, "strength": "STRONG",
                                "zscore": zscore, "short": short_median,
                                "long": long_median, "hurst": h60})
            elif confluence:
                print(f"\n    >>> DUAL CONFLUENCE: Z-score + Dual-horizon <<<")
                print(f"    >>> MODERATE {direction} SIGNAL")
                signals.append({"type": direction, "strength": "MODERATE",
                                "zscore": zscore, "short": short_median,
                                "long": long_median})
            else:
                print(f"\n    No confluence. Horizons don't align for {direction}.")
                if direction == "LONG" and not short_ok:
                    print(f"    Short-term still predicts DOWN -- wait for bounce")
                elif direction == "SHORT" and not short_ok:
                    print(f"    Short-term still predicts UP -- wait for rollover")

    if not signals:
        print(f"\n  No actionable confluence signals right now.")
    else:
        print(f"\n  Active signals: {len(signals)}")
        for s in signals:
            print(f"    {s['strength']} {s['type']}: z={s['zscore']:+.2f} "
                  f"short={s['short']:+.1f}% long={s['long']:+.1f}%")

    print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="LSTM + Quantamental Predictor")
    subs = parser.add_subparsers(dest="command")

    p_feat = subs.add_parser("build-features", help="Build features")
    p_feat.add_argument("--asset", required=True)

    p_train = subs.add_parser("train", help="Train models")
    p_train.add_argument("--asset", required=True)

    p_pred = subs.add_parser("predict", help="Predictions")
    p_pred.add_argument("--asset", required=True)

    p_conf = subs.add_parser("confluence", help="Mean-reversion confluence signals")
    p_conf.add_argument("--asset", required=True)
    p_conf.add_argument("--zscore", type=float, default=2.0)

    subs.add_parser("markets", help="Compare vs Polymarket")

    p_bt = subs.add_parser("backtest", help="Walk-forward backtest")
    p_bt.add_argument("--asset", required=True)

    args = parser.parse_args()

    dispatch = {
        "build-features": cmd_build_features,
        "train": cmd_train,
        "predict": cmd_predict,
        "confluence": cmd_confluence,
        "markets": cmd_markets,
        "backtest": cmd_backtest,
    }

    if args.command in dispatch:
        dispatch[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
