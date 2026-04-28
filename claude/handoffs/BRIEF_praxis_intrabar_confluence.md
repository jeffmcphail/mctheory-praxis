# Implementation Brief: Intrabar Multi-Horizon Confluence System

**Series:** praxis
**Priority:** P0 (primary trading signal system)
**Estimated Scope:** L (Large -- 4-6 hours)
**Date:** 2026-04-21

---

## Context

The daily-timescale multi-horizon LSTM (already built in `engines/lstm_predictor.py`) trained successfully but showed directional accuracy below coin-flip at every horizon (30-47%). Coverage intervals were well-calibrated (83% at 1d), but daily price direction is essentially unpredictable with standard features -- consistent with our TRADING_ATLAS finding that standard TA signals have no persistent edge on daily bars.

The mean-reversion edge lives in **intrabar timescales** where microstructure effects (funding rate spikes, liquidation cascades, z-score extremes) create short-lived predictable moves. The 1-minute data collector has been running since April 2026 and should now have substantial history accumulated beyond the Binance 30-day window.

Jeff's core insight: pairs trading mean-reversion fails when you enter too early (spread widens past stop) or too late (reversion already happened). A dual-horizon LSTM filter solves this -- only enter when:
1. Z-score threshold hit (spread is extended)
2. Short-term (5-min) predicts turn starting (momentum exhaustion)
3. Long-term (15-min) confirms reversion continues (structural confirmation)

This filter should reject 90%+ of raw z-score signals, keeping only high-conviction entries where all three conditions align.

---

## Objective

Build an intrabar version of the multi-horizon quantile LSTM that operates on 1-minute data, with horizons measured in bars (1/3/5/10/15) instead of days. Generate confluence signals using z-score + dual-horizon alignment + Hurst regime confirmation.

---

## Detailed Spec

### New File: `engines/intrabar_predictor.py`

This is largely a fork of `engines/lstm_predictor.py` adapted for intrabar timescales. DO NOT modify the existing daily predictor -- they run in parallel, serving different strategies.

### Architecture Changes from Daily Version

| Parameter | Daily (existing) | Intrabar (new) |
|-----------|------------------|----------------|
| `SEQUENCE_LENGTH` | 60 days | 60 bars (1 hour of 1-min data) |
| `MULTI_HORIZONS` | [1, 3, 7, 14, 30] days | [1, 3, 5, 10, 15] bars (minutes) |
| `QUANTILES` | [0.10, 0.25, 0.50, 0.75, 0.90] | Same |
| Data source | `ohlcv_daily` table | `ohlcv_1m` table |
| Features available | F&G, funding, on-chain, Hurst | OHLCV + Hurst only (no F&G/funding at 1-min resolution) |
| Hurst window | 30-day rolling | 30-bar rolling |

### Key Components to Build

**1. Data loader** -- `load_1m_data(asset, limit_bars=None)`:
```python
def load_1m_data(asset, limit_bars=None):
    """Load 1-minute OHLCV from ohlcv_1m table.

    Returns list of records sorted by timestamp ascending.
    Each record: {timestamp, datetime, open, high, low, close, volume}
    """
    conn = sqlite3.connect(CRYPTO_DB_PATH)
    conn.row_factory = sqlite3.Row
    query = """
        SELECT timestamp, datetime, open, high, low, close, volume
        FROM ohlcv_1m
        WHERE asset = ?
        ORDER BY timestamp ASC
    """
    if limit_bars:
        query += f" LIMIT {limit_bars}"
    rows = conn.execute(query, (asset,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]
```

**2. Feature engineering** -- simpler than daily version:
```python
def compute_intrabar_features(records, idx):
    """Compute intrabar features at index idx (backward-looking only).

    Features:
    - Multi-lookback returns (1, 5, 15, 30, 60 bars)
    - Price vs moving averages (5, 15, 30, 60 bars)
    - Realized volatility (15, 30, 60 bars)
    - RSI-14 (computed on 1-min bars)
    - Bollinger band position (20-bar)
    - Volume ratio (current vs 30-bar mean)
    - Multi-timescale Hurst (30, 60, 120 bar windows)
    - Rolling z-score (20, 60, 120 bar windows)  # NEW - key for mean reversion
    """
```

**3. Sequence builder** -- `build_intrabar_sequences(records)`:
- Input channels: [close, high, low, volume, hurst_30bar] (5 channels, no F&G/funding at this timescale)
- Sequence length: 60 bars
- Returns multi-horizon labels for all MULTI_HORIZONS at once

**4. Model** -- `IntrabarQuantileLSTM`:
- Same architecture as daily `MultiHorizonQuantileLSTM`
- Input size = 5 (vs 7 for daily)
- Hidden size = 96, 2 layers, dropout 0.3
- Per-horizon heads outputting 5 quantiles each
- Pinball (quantile) loss

**5. Training function** -- `train_intrabar_lstm()`:
- Use `time.time()`-based random seed for reproducibility
- Train/test split: 80/20 time-ordered (no shuffle)
- Batch size 64 (vs 32 for daily -- more data available)
- Adam optimizer, lr=0.001, weight_decay=1e-4
- ReduceLROnPlateau scheduler
- Early stopping with patience=20
- Max 150 epochs

**6. Confluence command** -- adapted for intrabar:
```python
def cmd_intrabar_confluence(args):
    """Mean-reversion confluence signals using 5-bar vs 15-bar prediction.

    Signal logic:
    1. Compute rolling z-score on 60-bar window
    2. If |z-score| < threshold: no signal
    3. Get LSTM multi-horizon prediction (latest 60 bars -> next 15 bar predictions)
    4. Check short (5-bar) vs long (15-bar) alignment

    For LONG (z-score <= -threshold, price is cheap):
      - Short-term (5-bar) median > 0 = bounce starting
      - Long-term (15-bar) median > 0 = reversion has room
      - Hurst 60-bar < 0.45 = mean-reverting regime confirmed

    For SHORT (z-score >= +threshold, price is rich):
      - Short-term (5-bar) median < 0 = rollover starting
      - Long-term (15-bar) median < 0 = reversion continuing
    """
```

### CLI Commands

```python
# Standard pipeline
python -m engines.intrabar_predictor build-features --asset BTC
python -m engines.intrabar_predictor train --asset BTC
python -m engines.intrabar_predictor predict --asset BTC
python -m engines.intrabar_predictor confluence --asset BTC
python -m engines.intrabar_predictor confluence --asset BTC --zscore 1.5

# Backtest: how would this have performed historically?
python -m engines.intrabar_predictor backtest --asset BTC --zscore 2.0
```

### Model Storage

```
models/intrabar/
+-- BTC_features.joblib
+-- BTC_multi_horizon.joblib
+-- BTC_multi_horizon_lstm.pt
+-- BTC_quant_mh_models.joblib
+-- ETH_* (same set)
```

Use a separate subdirectory from the daily `models/lstm/` to avoid collision.

---

## First-Pass Code (Skeleton)

```python
"""
engines/intrabar_predictor.py -- Intrabar multi-horizon confluence system

Operates on 1-minute OHLCV data. Predicts return distributions at
1/3/5/10/15 minute horizons. Generates mean-reversion confluence signals
by combining z-score + dual-horizon LSTM alignment + Hurst regime.

Usage:
    python -m engines.intrabar_predictor build-features --asset BTC
    python -m engines.intrabar_predictor train --asset BTC
    python -m engines.intrabar_predictor confluence --asset BTC
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

SEQUENCE_LENGTH = 60  # 60 one-minute bars = 1 hour lookback
MULTI_HORIZONS = [1, 3, 5, 10, 15]  # Bars (minutes)
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
TRAIN_SPLIT = 0.8

SUPPORTED_ASSETS = {"BTC", "ETH"}


# ===================================================================
# DATA LOADING
# ===================================================================
def load_1m_data(asset, limit_bars=None):
    """Load 1-minute OHLCV records sorted by timestamp ascending."""
    # ... implementation ...


# ===================================================================
# HURST ESTIMATION (reused from lstm_predictor.py)
# ===================================================================
def estimate_hurst(prices, min_window=10):
    """Rescaled Range (R/S) analysis -- same as daily predictor."""
    # ... copy from lstm_predictor.py ...


# ===================================================================
# FEATURE ENGINEERING
# ===================================================================
def compute_intrabar_features(records, idx):
    """Compute features at bar index idx."""
    # ... implementation ...


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
    return (arr[-1] - mean) / std


# ===================================================================
# SEQUENCE BUILDER
# ===================================================================
def build_intrabar_sequences(records):
    """Build sequences with multi-horizon return labels.

    Returns:
        sequences: (N, 60, 5) input arrays
        returns: (N, 5) actual future returns per horizon in MULTI_HORIZONS
        dates: list of datetime strings
    """
    # ... implementation ...


# ===================================================================
# MODEL
# ===================================================================
# Reuse MultiHorizonQuantileLSTM from lstm_predictor.py
# (Consider extracting it to a shared module later)


# ===================================================================
# TRAINING
# ===================================================================
def train_intrabar_lstm(sequences, returns, asset):
    """Train intrabar multi-horizon quantile LSTM.

    Same architecture as daily version but operating on minute bars.
    Uses pinball loss. Reports directional accuracy and p10-p90 coverage
    per horizon.
    """
    # ... implementation ...


def train_intrabar_xgboost(feature_rows, asset, horizons=None):
    """Train XGBoost quantile regression per horizon."""
    # ... implementation ...


# ===================================================================
# COMMANDS
# ===================================================================
def cmd_build_features(args):
    """Build features from 1-minute data."""
    # ... implementation ...


def cmd_train(args):
    """Train both LSTM and XGBoost multi-horizon models."""
    # ... implementation ...


def cmd_confluence(args):
    """Generate intrabar mean-reversion confluence signals."""
    # ... implementation ...


def cmd_backtest(args):
    """Walk-forward backtest of confluence signals.

    For each historical bar:
    1. Check if z-score threshold hit
    2. If yes, run LSTM prediction on preceding 60 bars
    3. Check dual-horizon confluence
    4. If signal, record hypothetical entry and track for 15 bars forward
    5. Compute hit rate, MAE, P&L distribution

    This is the validation step -- confirms the edge before live trading.
    """
    # ... implementation ...
```

---

## Acceptance Criteria

- [ ] `python -m engines.intrabar_predictor build-features --asset BTC` completes successfully
- [ ] Feature file saved to `models/intrabar/BTC_features.joblib`
- [ ] `python -m engines.intrabar_predictor train --asset BTC` completes successfully
- [ ] Training produces directional accuracy metrics per horizon
- [ ] p10-p90 coverage should be in 70-85% range (well-calibrated)
- [ ] `python -m engines.intrabar_predictor confluence --asset BTC` runs end-to-end
- [ ] Confluence output shows current z-score and multi-horizon predictions
- [ ] When z-score is within threshold, reports "no signal"
- [ ] When z-score exceeds threshold, checks dual-horizon alignment
- [ ] `python -m engines.intrabar_predictor backtest --asset BTC` produces historical signal analysis
- [ ] Report: total signals generated, hit rate, average R:R, equity curve
- [ ] All files pass ASCII-only check (no em dashes, emoji, box-drawing)
- [ ] All files pass `python -c "import ast; ast.parse(open('file.py').read())"`

---

## Known Pitfalls

**Data quality:**
- 1-minute data has gaps (exchange downtime, network issues). Filter out bars where `volume == 0` or timestamps have gaps > 2 minutes. Document the filtering in the retro.
- Binance retains only ~30 days of 1-minute data. Beyond that we accumulated via scheduled task -- may have gaps in early April 2026 when we first started collecting.

**Feature design:**
- No Fear & Greed at 1-min resolution -- it's a daily metric. Don't try to map it.
- No on-chain data at 1-min -- blockchain.info provides daily aggregates only.
- Funding rates update every 8 hours on Binance -- treat as a slowly-changing feature if included.
- Hurst estimation on 30 bars is noisy. Consider using longer windows (60+) for the feature itself even if the rolling update window is 30.

**Model training:**
- With 86K+ bars per asset, training is significantly slower than daily (which had ~800 sequences). Expect 30+ minutes on CPU.
- Batch size 64 is a reasonable starting point. Monitor memory usage.
- The `time.time()` seeding gives non-reproducible results across runs. If reproducibility matters, fix the seed.

**Confluence logic:**
- The 5-bar horizon is extremely noisy. The LSTM may predict tiny moves (+/- 0.02%) that are below transaction costs. Add a minimum predicted magnitude filter.
- Z-score thresholds that work on daily bars (+/-2.0) may be too aggressive or too lax on 1-min. Backtest to find the right threshold.

**ASCII compliance:**
- This script will likely be wrapped by a scheduled task eventually. Must be ASCII-only from the start.
- Use `-` and `--` instead of em dashes (--).
- Use `+--+` for table borders, not box-drawing chars.

**Backtest gotchas:**
- Walk-forward only. No peeking at future data.
- Account for transaction costs (Polymarket ~2%, Binance ~0.1%).
- Track slippage estimate (assume 0.05% per side).
- Report results both with and without costs.

---

## Dependencies

- PyTorch (already installed: `torch==2.11.0+cpu`)
- XGBoost (already installed)
- NumPy (already installed)
- python-dotenv (already installed)

No new dependencies required.

---

## References

- Chat session: praxis_main_current (2026-04-21)
- Existing file: `engines/lstm_predictor.py` (daily version, 1669 lines)
- Data source: `data/crypto_data.db` -> `ohlcv_1m` table (86K+ rows as of initial backfill)
- Design inspiration: Volterra framework (multi-timescale Hurst), Chan/Burgess pairs trading CPO
- Jeff's core insight: dual-horizon filter solves the "entered too early vs too late" problem in mean-reversion trading

---

## Open Questions (OK to decide during implementation)

1. Should we build the XGBoost quantile version too, or just the LSTM? (Recommendation: both, same as daily version -- XGBoost provides a second opinion on distribution shape)
2. Should Hurst be computed on 1-min bars (noisy) or on a resampled higher-timescale (5-min)? (Recommendation: try 60-bar Hurst first, fall back to 5-min resampled if too noisy)
3. What's the right minimum magnitude filter for signal quality? (Recommendation: |predicted median| > 0.05% as starting point, tune via backtest)

These can be decided by Claude Code during implementation. Document the choice in the retro.
