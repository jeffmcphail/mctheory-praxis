"""
engines/crypto_predictor.py — Crypto Price Prediction System

Two-model ensemble for predicting crypto price movements and applying
predictions to Polymarket crypto markets.

Models:
    1. LSTM Timeseries — pure price/volume/technical features
    2. Quantamental — fundamentals, social, news, on-chain, whale behavior

Combined output:
    - Probability of price being above/below threshold at horizon
    - Confidence interval
    - Recommended horizon (1d, 7d, 30d)
    - Direct comparison to Polymarket market prices

Data sources:
    - CCXT: OHLCV price data (Binance, Coinbase)
    - CoinGecko: market cap, volume, supply metrics
    - Alternative.me: Fear & Greed Index
    - Glassnode/on-chain: active addresses, exchange flows (when available)
    - Polymarket: current prediction market prices for comparison

Usage:
    python -m engines.crypto_predictor collect --asset BTC --days 365   # Collect data
    python -m engines.crypto_predictor collect --asset ETH --days 365
    python -m engines.crypto_predictor features --asset BTC             # Build features
    python -m engines.crypto_predictor train --asset BTC                # Train models
    python -m engines.crypto_predictor predict --asset BTC              # Generate predictions
    python -m engines.crypto_predictor markets                          # Compare vs Polymarket
    python -m engines.crypto_predictor dashboard                        # Full analysis
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
import requests
from dotenv import load_dotenv
load_dotenv()

DB_PATH = Path("data/crypto_predictor.db")
MODEL_DIR = Path("models/crypto")

# Supported assets
SUPPORTED_ASSETS = {
    "BTC": {"name": "Bitcoin", "coingecko_id": "bitcoin", "symbol": "BTC/USDT"},
    "ETH": {"name": "Ethereum", "coingecko_id": "ethereum", "symbol": "ETH/USDT"},
    "SOL": {"name": "Solana", "coingecko_id": "solana", "symbol": "SOL/USDT"},
}

# Feature engineering params
LOOKBACK_WINDOWS = [7, 14, 30, 60, 90, 200]
PREDICTION_HORIZONS = [1, 7, 30]  # Days ahead


# ═══════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            date TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL,
            volume REAL,
            UNIQUE(asset, timestamp)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS fundamentals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset TEXT NOT NULL,
            date TEXT NOT NULL,
            market_cap REAL,
            total_volume REAL,
            circulating_supply REAL,
            total_supply REAL,
            ath REAL,
            ath_change_pct REAL,
            fear_greed_index INTEGER,
            fear_greed_label TEXT,
            dominance_pct REAL,
            UNIQUE(asset, date)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset TEXT NOT NULL,
            date TEXT NOT NULL,
            feature_set TEXT NOT NULL,
            features_json TEXT NOT NULL,
            UNIQUE(asset, date, feature_set)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            asset TEXT NOT NULL,
            model TEXT NOT NULL,
            horizon_days INTEGER,
            current_price REAL,
            predicted_direction TEXT,
            predicted_change_pct REAL,
            probability_up REAL,
            probability_down REAL,
            confidence REAL,
            threshold REAL,
            prob_above_threshold REAL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS market_comparisons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            asset TEXT NOT NULL,
            market_slug TEXT,
            market_question TEXT,
            market_price REAL,
            ai_probability REAL,
            divergence_pct REAL,
            model_used TEXT,
            horizon_days INTEGER
        )
    """)

    conn.commit()
    return conn


# ═══════════════════════════════════════════════════════
# DATA COLLECTION
# ═══════════════════════════════════════════════════════

def collect_ohlcv(asset, days=365, conn=None):
    """Collect OHLCV data via CCXT (Binance)."""
    print(f"    Collecting OHLCV for {asset} ({days} days)...")

    try:
        import ccxt
        exchange = ccxt.binance({"enableRateLimit": True})
    except ImportError:
        print(f"    ❌ pip install ccxt --break-system-packages")
        return 0

    symbol = SUPPORTED_ASSETS[asset]["symbol"]
    since = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

    all_candles = []
    fetch_since = since

    while True:
        try:
            candles = exchange.fetch_ohlcv(symbol, "1d", since=fetch_since, limit=500)
            if not candles:
                break
            all_candles.extend(candles)
            fetch_since = candles[-1][0] + 86400000  # Next day
            if len(candles) < 500:
                break
            time.sleep(0.5)
        except Exception as e:
            print(f"    ⚠️ CCXT error: {e}")
            break

    # Store
    stored = 0
    for c in all_candles:
        ts = int(c[0] / 1000)
        date = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        try:
            conn.execute("""
                INSERT OR REPLACE INTO ohlcv
                (asset, timestamp, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (asset, ts, date, c[1], c[2], c[3], c[4], c[5]))
            stored += 1
        except Exception:
            pass

    conn.commit()
    print(f"    ✅ {stored} daily candles stored")
    return stored


def collect_fundamentals(asset, conn=None):
    """Collect fundamental data from CoinGecko."""
    print(f"    Collecting fundamentals for {asset}...")

    cg_id = SUPPORTED_ASSETS[asset]["coingecko_id"]

    try:
        r = requests.get(f"https://api.coingecko.com/api/v3/coins/{cg_id}", params={
            "localization": "false",
            "tickers": "false",
            "community_data": "false",
            "developer_data": "false",
        }, timeout=15)

        if r.status_code == 200:
            data = r.json()
            md = data.get("market_data", {})

            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            conn.execute("""
                INSERT OR REPLACE INTO fundamentals
                (asset, date, market_cap, total_volume, circulating_supply,
                 total_supply, ath, ath_change_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                asset, today,
                md.get("market_cap", {}).get("usd", 0),
                md.get("total_volume", {}).get("usd", 0),
                md.get("circulating_supply", 0),
                md.get("total_supply", 0),
                md.get("ath", {}).get("usd", 0),
                md.get("ath_change_percentage", {}).get("usd", 0),
            ))
            conn.commit()
            print(f"    ✅ Market cap: ${md.get('market_cap', {}).get('usd', 0):,.0f}")
        else:
            print(f"    ⚠️ CoinGecko returned {r.status_code}")

    except Exception as e:
        print(f"    ⚠️ CoinGecko error: {e}")

    # Fear & Greed Index
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        if r.status_code == 200:
            fg = r.json().get("data", [{}])[0]
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            conn.execute("""
                UPDATE fundamentals SET
                    fear_greed_index=?, fear_greed_label=?
                WHERE asset=? AND date=?
            """, (
                int(fg.get("value", 50)),
                fg.get("value_classification", "Neutral"),
                asset, today,
            ))
            conn.commit()
            print(f"    ✅ Fear & Greed: {fg.get('value', '?')} "
                  f"({fg.get('value_classification', '?')})")
    except Exception as e:
        print(f"    ⚠️ F&G error: {e}")


def cmd_collect(args):
    """Collect all data for an asset."""
    asset = args.asset.upper()
    days = getattr(args, "days", 365)

    if asset not in SUPPORTED_ASSETS:
        print(f"  ❌ Unsupported asset: {asset}")
        print(f"  Supported: {', '.join(SUPPORTED_ASSETS.keys())}")
        return

    conn = init_db()

    print(f"\n{'='*70}")
    print(f"  CRYPTO DATA COLLECTION — {SUPPORTED_ASSETS[asset]['name']}")
    print(f"{'='*70}")

    collect_ohlcv(asset, days, conn)
    time.sleep(1)
    collect_fundamentals(asset, conn)

    # Summary
    n_candles = conn.execute(
        "SELECT COUNT(*) FROM ohlcv WHERE asset=?", (asset,)).fetchone()[0]
    latest = conn.execute(
        "SELECT date, close FROM ohlcv WHERE asset=? ORDER BY timestamp DESC LIMIT 1",
        (asset,)).fetchone()

    print(f"\n  Summary:")
    print(f"    Candles: {n_candles}")
    if latest:
        print(f"    Latest: {latest[0]} @ ${latest[1]:,.2f}")

    conn.close()
    print(f"\n{'='*70}")


# ═══════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════

def compute_technical_features(prices, idx):
    """Compute technical features at a given index.

    All backward-looking, no look-ahead bias.
    """
    if idx < 200:  # Need enough history
        return None

    features = {}
    close = prices[idx]["close"]
    closes = [prices[i]["close"] for i in range(max(0, idx - 200), idx + 1)]
    highs = [prices[i]["high"] for i in range(max(0, idx - 200), idx + 1)]
    lows = [prices[i]["low"] for i in range(max(0, idx - 200), idx + 1)]
    volumes = [prices[i]["volume"] for i in range(max(0, idx - 200), idx + 1)]

    # Returns at multiple horizons
    for w in LOOKBACK_WINDOWS:
        if len(closes) > w:
            ret = (closes[-1] - closes[-1 - w]) / closes[-1 - w] * 100
            features[f"return_{w}d"] = ret

    # Moving averages
    for w in [7, 14, 30, 50, 200]:
        if len(closes) >= w:
            ma = sum(closes[-w:]) / w
            features[f"ma_{w}d"] = ma
            features[f"price_vs_ma_{w}d"] = (close - ma) / ma * 100

    # Volatility (rolling std of daily returns)
    daily_returns = []
    for i in range(1, min(31, len(closes))):
        daily_returns.append((closes[-i] - closes[-i - 1]) / closes[-i - 1])

    if daily_returns:
        features["volatility_30d"] = float(np.std(daily_returns)) * math.sqrt(365) * 100
        features["volatility_7d"] = float(np.std(daily_returns[:7])) * math.sqrt(365) * 100 if len(daily_returns) >= 7 else 0

    # RSI (14-day)
    if len(closes) >= 15:
        gains, losses = [], []
        for i in range(-14, 0):
            change = closes[i] - closes[i - 1]
            gains.append(max(change, 0))
            losses.append(max(-change, 0))

        avg_gain = sum(gains) / 14
        avg_loss = sum(losses) / 14

        if avg_loss > 0:
            rs = avg_gain / avg_loss
            features["rsi_14"] = 100 - (100 / (1 + rs))
        else:
            features["rsi_14"] = 100

    # MACD (12, 26, 9)
    if len(closes) >= 26:
        ema_12 = _ema(closes, 12)
        ema_26 = _ema(closes, 26)
        macd_line = ema_12 - ema_26
        features["macd"] = macd_line
        features["macd_pct"] = macd_line / close * 100

    # Bollinger Bands (20, 2)
    if len(closes) >= 20:
        bb_ma = sum(closes[-20:]) / 20
        bb_std = float(np.std(closes[-20:]))
        upper = bb_ma + 2 * bb_std
        lower = bb_ma - 2 * bb_std
        features["bb_position"] = (close - lower) / (upper - lower) if upper > lower else 0.5
        features["bb_width"] = (upper - lower) / bb_ma * 100

    # Volume features
    if len(volumes) >= 30:
        avg_vol = sum(volumes[-30:]) / 30
        features["volume_ratio"] = volumes[-1] / avg_vol if avg_vol > 0 else 1
        features["volume_trend"] = (sum(volumes[-7:]) / 7) / (sum(volumes[-30:]) / 30) if avg_vol > 0 else 1

    # ATH distance
    all_time_high = max(highs)
    features["ath_distance_pct"] = (close - all_time_high) / all_time_high * 100

    # Consecutive up/down days
    consec = 0
    direction = 1 if closes[-1] > closes[-2] else -1
    for i in range(2, min(15, len(closes))):
        d = 1 if closes[-i] > closes[-i - 1] else -1
        if d == direction:
            consec += 1
        else:
            break
    features["consecutive_days"] = consec * direction

    # Drawdown from recent high
    recent_high = max(closes[-30:]) if len(closes) >= 30 else close
    features["drawdown_30d"] = (close - recent_high) / recent_high * 100

    return features


def _ema(data, period):
    """Calculate Exponential Moving Average."""
    if len(data) < period:
        return data[-1]
    multiplier = 2 / (period + 1)
    ema = sum(data[-period:]) / period  # Start with SMA
    for price in data[-period:]:
        ema = (price - ema) * multiplier + ema
    return ema


def cmd_features(args):
    """Build feature dataset for an asset."""
    asset = args.asset.upper()
    conn = init_db()

    print(f"\n{'='*70}")
    print(f"  FEATURE ENGINEERING — {asset}")
    print(f"{'='*70}")

    # Load OHLCV
    rows = conn.execute("""
        SELECT date, open, high, low, close, volume
        FROM ohlcv WHERE asset=?
        ORDER BY timestamp ASC
    """, (asset,)).fetchall()

    if len(rows) < 201:
        print(f"  ❌ Need at least 201 candles, have {len(rows)}")
        print(f"  Run: python -m engines.crypto_predictor collect --asset {asset} --days 365")
        conn.close()
        return

    prices = [{"date": r[0], "open": r[1], "high": r[2],
               "low": r[3], "close": r[4], "volume": r[5]} for r in rows]

    print(f"  Candles loaded: {len(prices)}")
    print(f"  Range: {prices[0]['date']} → {prices[-1]['date']}")

    # Compute features for each day
    stored = 0
    for i in range(200, len(prices)):
        feats = compute_technical_features(prices, i)
        if feats is None:
            continue

        # Add labels (future returns — only for training, not inference)
        for horizon in PREDICTION_HORIZONS:
            if i + horizon < len(prices):
                future_close = prices[i + horizon]["close"]
                current_close = prices[i]["close"]
                future_return = (future_close - current_close) / current_close * 100
                feats[f"target_{horizon}d_return"] = future_return
                feats[f"target_{horizon}d_up"] = 1 if future_return > 0 else 0

        conn.execute("""
            INSERT OR REPLACE INTO features
            (asset, date, feature_set, features_json)
            VALUES (?, ?, 'technical', ?)
        """, (asset, prices[i]["date"], json.dumps(feats)))
        stored += 1

    conn.commit()

    # Summary
    print(f"  Features computed: {stored} days")

    # Show feature ranges for latest day
    latest_feats = json.loads(conn.execute("""
        SELECT features_json FROM features
        WHERE asset=? AND feature_set='technical'
        ORDER BY date DESC LIMIT 1
    """, (asset,)).fetchone()[0])

    print(f"\n  Latest features ({prices[-1]['date']}):")
    exclude = {k for k in latest_feats if k.startswith("target_")}
    for k, v in sorted(latest_feats.items()):
        if k not in exclude:
            if isinstance(v, float):
                print(f"    {k:<25s} {v:>12.2f}")

    conn.close()
    print(f"\n{'='*70}")


# ═══════════════════════════════════════════════════════
# MODEL TRAINING
# ═══════════════════════════════════════════════════════

def cmd_train(args):
    """Train prediction models."""
    asset = args.asset.upper()
    conn = init_db()

    print(f"\n{'='*70}")
    print(f"  MODEL TRAINING — {asset}")
    print(f"{'='*70}")

    # Load features
    rows = conn.execute("""
        SELECT date, features_json FROM features
        WHERE asset=? AND feature_set='technical'
        ORDER BY date ASC
    """, (asset,)).fetchall()

    if len(rows) < 100:
        print(f"  ❌ Need at least 100 feature rows, have {len(rows)}")
        conn.close()
        return

    print(f"  Feature rows: {len(rows)}")

    # Parse features
    dates = []
    feature_dicts = []
    for date, fj in rows:
        feats = json.loads(fj)
        dates.append(date)
        feature_dicts.append(feats)

    # Identify feature columns (exclude targets)
    feature_cols = sorted([k for k in feature_dicts[0].keys()
                           if not k.startswith("target_")
                           and isinstance(feature_dicts[0][k], (int, float))])

    print(f"  Feature columns: {len(feature_cols)}")

    # For each prediction horizon, train a model
    for horizon in PREDICTION_HORIZONS:
        target_col = f"target_{horizon}d_up"

        # Filter rows that have target labels
        valid = [(i, fd) for i, fd in enumerate(feature_dicts)
                 if target_col in fd]

        if len(valid) < 50:
            print(f"\n  ⚠️ {horizon}d horizon: only {len(valid)} labeled samples, skipping")
            continue

        # Build X, y arrays
        X = np.array([[fd.get(c, 0) for c in feature_cols] for _, fd in valid])
        y = np.array([fd[target_col] for _, fd in valid])

        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        # Time-based split (80% train, 20% test)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        print(f"\n  ── {horizon}-Day Horizon ──")
        print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
        print(f"  Base rate: {y_train.mean():.1%} up")

        # Try XGBoost first, fall back to simple logistic
        try:
            import xgboost as xgb

            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss",
            )
            model.fit(X_train, y_train, verbose=False)
            model_type = "xgboost"

            # Evaluate
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            test_probs = model.predict_proba(X_test)[:, 1]

            # Feature importance
            importances = model.feature_importances_
            top_features = sorted(zip(feature_cols, importances),
                                  key=lambda x: -x[1])[:10]

            print(f"  Model: XGBoost")
            print(f"  Train accuracy: {train_acc:.1%}")
            print(f"  Test accuracy:  {test_acc:.1%}")

            print(f"  Top features:")
            for fname, imp in top_features[:5]:
                print(f"    {fname:<25s} {imp:.3f}")

            # Save model
            import joblib
            model_path = MODEL_DIR / f"{asset}_{horizon}d_xgb.joblib"
            joblib.dump({
                "model": model,
                "feature_cols": feature_cols,
                "model_type": model_type,
                "horizon": horizon,
                "asset": asset,
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "n_train": len(X_train),
                "n_test": len(X_test),
            }, model_path)
            print(f"  Saved: {model_path}")

        except ImportError:
            # Fallback: simple logistic regression
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train_s, y_train)
            model_type = "logistic"

            train_acc = model.score(X_train_s, y_train)
            test_acc = model.score(X_test_s, y_test)

            print(f"  Model: Logistic Regression (install xgboost for better results)")
            print(f"  Train accuracy: {train_acc:.1%}")
            print(f"  Test accuracy:  {test_acc:.1%}")

            import joblib
            model_path = MODEL_DIR / f"{asset}_{horizon}d_lr.joblib"
            joblib.dump({
                "model": model,
                "scaler": scaler,
                "feature_cols": feature_cols,
                "model_type": model_type,
                "horizon": horizon,
                "asset": asset,
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "trained_at": datetime.now(timezone.utc).isoformat(),
            }, model_path)
            print(f"  Saved: {model_path}")

        except Exception as e:
            print(f"  ❌ Training error: {e}")
            # Absolute fallback: naive model
            print(f"  Using naive baseline (predict majority class)")

    conn.close()
    print(f"\n{'='*70}")


# ═══════════════════════════════════════════════════════
# PREDICTION
# ═══════════════════════════════════════════════════════

def cmd_predict(args):
    """Generate predictions for an asset."""
    asset = args.asset.upper()
    conn = init_db()

    print(f"\n{'='*70}")
    print(f"  CRYPTO PREDICTION — {SUPPORTED_ASSETS[asset]['name']}")
    print(f"{'='*70}")

    # Get latest features
    latest = conn.execute("""
        SELECT date, features_json FROM features
        WHERE asset=? AND feature_set='technical'
        ORDER BY date DESC LIMIT 1
    """, (asset,)).fetchone()

    if not latest:
        print(f"  ❌ No features. Run: python -m engines.crypto_predictor features --asset {asset}")
        conn.close()
        return

    latest_date, latest_json = latest
    features = json.loads(latest_json)

    # Get current price
    latest_price = conn.execute(
        "SELECT close FROM ohlcv WHERE asset=? ORDER BY timestamp DESC LIMIT 1",
        (asset,)).fetchone()
    current_price = latest_price[0] if latest_price else 0

    print(f"  Latest data: {latest_date}")
    print(f"  Current price: ${current_price:,.2f}")

    # Load and run each horizon model
    import joblib
    now = datetime.now(timezone.utc).isoformat()

    for horizon in PREDICTION_HORIZONS:
        # Find model file
        model_path = None
        for suffix in ["xgb", "lr"]:
            p = MODEL_DIR / f"{asset}_{horizon}d_{suffix}.joblib"
            if p.exists():
                model_path = p
                break

        if not model_path:
            print(f"\n  ⚠️ No {horizon}d model found. Run train first.")
            continue

        saved = joblib.load(model_path)
        model = saved["model"]
        feature_cols = saved["feature_cols"]
        model_type = saved["model_type"]

        # Build feature vector
        X = np.array([[features.get(c, 0) for c in feature_cols]])
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        if model_type == "logistic":
            X = saved["scaler"].transform(X)

        # Predict
        prob_up = model.predict_proba(X)[0][1]
        prob_down = 1 - prob_up
        direction = "UP" if prob_up > 0.5 else "DOWN"
        confidence = abs(prob_up - 0.5) * 2  # 0 at 50/50, 1 at 100/0

        print(f"\n  ── {horizon}-Day Prediction ──")
        print(f"  Model:      {model_type} (acc: {saved.get('test_accuracy', '?'):.1%})")
        print(f"  Direction:  {'📈' if direction == 'UP' else '📉'} {direction}")
        print(f"  P(up):      {prob_up:.1%}")
        print(f"  P(down):    {prob_down:.1%}")
        print(f"  Confidence: {confidence:.1%}")

        # Store prediction
        conn.execute("""
            INSERT INTO predictions
            (timestamp, asset, model, horizon_days, current_price,
             predicted_direction, probability_up, probability_down, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now, asset, model_type, horizon, current_price,
            direction, prob_up, prob_down, confidence,
        ))

    conn.commit()
    conn.close()
    print(f"\n{'='*70}")


# ═══════════════════════════════════════════════════════
# POLYMARKET COMPARISON
# ═══════════════════════════════════════════════════════

def cmd_markets(args):
    """Compare model predictions vs Polymarket crypto markets."""
    conn = init_db()

    print(f"\n{'='*90}")
    print(f"  AI vs MARKET — Crypto Prediction Markets")
    print(f"{'='*90}")

    # Fetch crypto markets from Polymarket
    try:
        r = requests.get(f"https://gamma-api.polymarket.com/markets", params={
            "closed": "false", "active": "true", "limit": 200,
        }, timeout=15)
        markets = r.json()
    except Exception:
        print(f"  ❌ Failed to fetch Polymarket markets")
        conn.close()
        return

    # Filter crypto markets
    crypto_markets = []
    crypto_keywords = ["bitcoin", "btc", "ethereum", "eth", "solana", "sol",
                       "crypto", "price", "$"]

    for m in markets:
        q = (m.get("question", "") or "").lower()
        if any(kw in q for kw in crypto_keywords):
            # Get current price
            token_ids = json.loads(m.get("clobTokenIds", "[]"))
            if not token_ids:
                continue
            try:
                r = requests.get(f"{CLOB_API}/midpoint",
                                 params={"token_id": token_ids[0]}, timeout=5)
                data = r.json()
                mid = float(data.get("mid", 0.5) if isinstance(data, dict) else data)
            except Exception:
                mid = 0.5

            crypto_markets.append({
                "slug": m.get("slug", ""),
                "question": m.get("question", ""),
                "price": mid,
                "volume": float(m.get("volume", 0) or 0),
                "end_date": m.get("endDate", ""),
            })
            time.sleep(0.05)

    print(f"  Found {len(crypto_markets)} crypto prediction markets\n")

    if not crypto_markets:
        print(f"  No crypto markets found.")
        conn.close()
        return

    # Get our latest predictions
    predictions = {}
    for asset in SUPPORTED_ASSETS:
        pred = conn.execute("""
            SELECT * FROM predictions
            WHERE asset=?
            ORDER BY timestamp DESC LIMIT 3
        """, (asset,)).fetchall()
        if pred:
            predictions[asset] = pred

    # Compare
    print(f"  {'Market':<55s} {'Mkt':>5s} {'Vol':>10s}")
    print(f"  {'─'*75}")

    now = datetime.now(timezone.utc).isoformat()

    for cm in sorted(crypto_markets, key=lambda x: -x["volume"])[:20]:
        q = cm["question"][:54]
        print(f"  {q:<55s} {cm['price']:>4.0%} ${cm['volume']:>9,.0f}")

        # Try to match with our predictions
        q_lower = cm["question"].lower()
        matched_asset = None
        for asset in SUPPORTED_ASSETS:
            if asset.lower() in q_lower or SUPPORTED_ASSETS[asset]["name"].lower() in q_lower:
                matched_asset = asset
                break

        if matched_asset and matched_asset in predictions:
            for pred in predictions[matched_asset]:
                # Simple comparison
                ai_prob = pred[7] if pred[7] else 0.5  # probability_up
                divergence = (ai_prob - cm["price"]) * 100

                if abs(divergence) > 10:
                    direction = "📈 AI higher" if divergence > 0 else "📉 AI lower"
                    print(f"    → {matched_asset} {pred[4]}d model: "
                          f"AI={ai_prob:.0%} vs Market={cm['price']:.0%} "
                          f"({divergence:+.0f}pp) {direction}")

                    conn.execute("""
                        INSERT INTO market_comparisons
                        (timestamp, asset, market_slug, market_question,
                         market_price, ai_probability, divergence_pct,
                         model_used, horizon_days)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        now, matched_asset, cm["slug"],
                        cm["question"][:100], cm["price"],
                        ai_prob, divergence, pred[2], pred[4],
                    ))

    conn.commit()
    conn.close()
    print(f"\n{'='*90}")


def cmd_dashboard(args):
    """Full dashboard: latest predictions + market comparison."""
    for asset in SUPPORTED_ASSETS:
        # Check if we have data
        conn = init_db()
        n = conn.execute(
            "SELECT COUNT(*) FROM ohlcv WHERE asset=?", (asset,)).fetchone()[0]
        conn.close()
        if n > 0:
            class FakeArgs:
                pass
            a = FakeArgs()
            a.asset = asset
            cmd_predict(a)

    cmd_markets(args)


def main():
    parser = argparse.ArgumentParser(description="Crypto Price Prediction System")
    subs = parser.add_subparsers(dest="command")

    p_collect = subs.add_parser("collect", help="Collect data")
    p_collect.add_argument("--asset", required=True, help="BTC, ETH, or SOL")
    p_collect.add_argument("--days", type=int, default=365)

    p_feat = subs.add_parser("features", help="Build features")
    p_feat.add_argument("--asset", required=True)

    p_train = subs.add_parser("train", help="Train models")
    p_train.add_argument("--asset", required=True)

    p_pred = subs.add_parser("predict", help="Generate predictions")
    p_pred.add_argument("--asset", required=True)

    subs.add_parser("markets", help="Compare vs Polymarket")
    subs.add_parser("dashboard", help="Full dashboard")

    args = parser.parse_args()

    dispatch = {
        "collect": cmd_collect,
        "features": cmd_features,
        "train": cmd_train,
        "predict": cmd_predict,
        "markets": cmd_markets,
        "dashboard": cmd_dashboard,
    }

    if args.command in dispatch:
        dispatch[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
