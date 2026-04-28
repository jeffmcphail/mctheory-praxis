"""
engines/spike_features.py — Feature Engineering for Event-Driven Spike Predictor

Transforms raw price history + spike detections into training datasets for:
  Stage 1: Spike Classifier (binary: is this a genuine Spike?)
  Stage 2: Peak Proximity Estimator (regression: 0.0-1.0, how far through the spike)

Architecture:
  1. Reads price_history + spikes + markets from spike_scanner.db
  2. For each price movement that crosses the trigger threshold:
     - Computes features observable at that moment (no look-ahead)
     - Labels it as Spike or Not-Spike (Stage 1, using hindsight)
     - If Spike: computes peak proximity label at each time step (Stage 2)
  3. Outputs training CSVs ready for XGBoost

Usage:
    python -m engines.spike_features build                    # Build training data
    python -m engines.spike_features build --threshold 5.0    # Trigger threshold
    python -m engines.spike_features stats                    # Show dataset stats
    python -m engines.spike_features inspect --slug <slug>    # Visualize one market's spikes
"""
import argparse
import csv
import json
import math
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DB_PATH = Path("data/spike_scanner.db")
OUTPUT_DIR = Path("data/training")

# ═══════════════════════════════════════════════════════
# SPIKE DEFINITION (hindsight labels)
# ═══════════════════════════════════════════════════════

# Stage 1 label: what qualifies as a "genuine Spike" in hindsight
SPIKE_MIN_MAGNITUDE = 8.0    # Minimum % move from baseline
SPIKE_MAX_DURATION = 60      # Must achieve peak within N minutes
SPIKE_MIN_REVERSION = 0.25   # Must retrace at least 25% of move after peak

# Trigger: when the live scanner would fire
TRIGGER_THRESHOLD = 5.0      # Scanner fires when move exceeds this %
TRIGGER_WINDOW = 60          # Within this many minutes

# Stage 2: sampling
STAGE2_SAMPLE_INTERVAL = 60  # Sample features every N seconds during a spike


# ═══════════════════════════════════════════════════════
# FEATURE COMPUTATION
# ═══════════════════════════════════════════════════════

def compute_price_features(prices, current_idx, slug=""):
    """Compute features observable at prices[current_idx].

    All features are backward-looking — no future information.

    Args:
        prices: list of (timestamp, price) tuples, sorted by time
        current_idx: index into prices for "now"

    Returns:
        dict of feature_name: value
    """
    if current_idx < 2:
        return None

    t_now, p_now = prices[current_idx]
    features = {}

    # ── Price at current time ──
    features["price"] = p_now
    features["timestamp"] = t_now

    # ── Returns at multiple lookback windows ──
    lookback_windows = [
        ("1m", 60), ("2m", 120), ("5m", 300),
        ("10m", 600), ("15m", 900), ("30m", 1800), ("60m", 3600)
    ]

    for name, secs in lookback_windows:
        # Find price at t_now - secs
        target_t = t_now - secs
        best_idx = None
        best_dist = float("inf")
        for j in range(current_idx, -1, -1):
            dist = abs(prices[j][0] - target_t)
            if dist < best_dist:
                best_dist = dist
                best_idx = j
            if prices[j][0] < target_t - 120:  # Stop searching beyond 2 min tolerance
                break

        if best_idx is not None and best_dist < 180:  # Within 3 min tolerance
            p_past = prices[best_idx][1]
            if p_past > 0:
                ret = (p_now - p_past) / p_past * 100
                features[f"return_{name}"] = ret
            else:
                features[f"return_{name}"] = 0
        else:
            features[f"return_{name}"] = 0

    # ── Velocity (rate of change, %/minute) ──
    # Use 2-minute window for smoothing
    if current_idx >= 2:
        lookback = 120  # 2 minutes
        target_t = t_now - lookback
        past_idx = current_idx
        for j in range(current_idx - 1, -1, -1):
            if prices[j][0] <= target_t:
                past_idx = j
                break

        t_past, p_past = prices[past_idx]
        dt_mins = max((t_now - t_past) / 60, 0.01)
        if p_past > 0:
            features["velocity"] = ((p_now - p_past) / p_past * 100) / dt_mins
        else:
            features["velocity"] = 0
    else:
        features["velocity"] = 0

    # ── Acceleration (change in velocity) ──
    if current_idx >= 4:
        # Velocity now vs velocity 2 min ago
        mid_idx = max(0, current_idx - 2)
        t_mid, p_mid = prices[mid_idx]

        # Find point 2 min before mid
        target_t2 = t_mid - 120
        past2_idx = mid_idx
        for j in range(mid_idx - 1, -1, -1):
            if prices[j][0] <= target_t2:
                past2_idx = j
                break

        t_past2, p_past2 = prices[past2_idx]

        dt1 = max((t_now - t_mid) / 60, 0.01)
        dt2 = max((t_mid - t_past2) / 60, 0.01)

        vel_now = ((p_now - p_mid) / max(p_mid, 0.001) * 100) / dt1
        vel_past = ((p_mid - p_past2) / max(p_past2, 0.001) * 100) / dt2

        features["acceleration"] = vel_now - vel_past
    else:
        features["acceleration"] = 0

    # ── Volatility (rolling std of 1-min returns) ──
    recent_returns = []
    for j in range(max(0, current_idx - 10), current_idx):
        if j > 0 and prices[j - 1][1] > 0:
            r = (prices[j][1] - prices[j - 1][1]) / prices[j - 1][1] * 100
            recent_returns.append(r)

    if len(recent_returns) >= 3:
        features["volatility_10"] = float(np.std(recent_returns))
    else:
        features["volatility_10"] = 0

    # ── Price position (where in 0-1 range) ──
    features["price_position"] = p_now  # Already 0-1 for Polymarket

    # ── Distance from extremes ──
    # Min/max price in last 30 min
    window_start = t_now - 1800
    window_prices = [p for t, p in prices[:current_idx + 1] if t >= window_start and p > 0]
    if window_prices:
        features["price_vs_30m_high"] = p_now / max(window_prices) if max(window_prices) > 0 else 1
        features["price_vs_30m_low"] = p_now / min(window_prices) if min(window_prices) > 0 else 1
        features["range_30m"] = (max(window_prices) - min(window_prices)) * 100
    else:
        features["price_vs_30m_high"] = 1
        features["price_vs_30m_low"] = 1
        features["range_30m"] = 0

    # ── Move magnitude from local baseline ──
    # Baseline = average price in the 5 minutes before the current 10-minute window
    baseline_start = t_now - 900  # 15 min ago
    baseline_end = t_now - 600    # 10 min ago
    baseline_prices = [p for t, p in prices[:current_idx + 1]
                       if baseline_start <= t <= baseline_end and p > 0]
    if baseline_prices:
        baseline = sum(baseline_prices) / len(baseline_prices)
        features["move_from_baseline"] = (p_now - baseline) / baseline * 100 if baseline > 0 else 0
        features["baseline_price"] = baseline
    else:
        features["move_from_baseline"] = 0
        features["baseline_price"] = p_now

    # ── Consecutive direction (how many ticks in same direction) ──
    consec = 0
    if current_idx >= 1:
        direction = 1 if p_now > prices[current_idx - 1][1] else -1
        for j in range(current_idx - 1, max(0, current_idx - 20), -1):
            if j > 0:
                d = 1 if prices[j][1] > prices[j - 1][1] else -1
                if d == direction:
                    consec += 1
                else:
                    break
    features["consecutive_direction"] = consec

    return features


def compute_metadata_features(market_row):
    """Compute static market metadata features.

    Args:
        market_row: dict from markets table

    Returns:
        dict of feature_name: value
    """
    features = {}

    # Event type (will be one-hot encoded later)
    features["event_type"] = market_row.get("event_type", "unknown")

    # Volume (log-transformed)
    vol = float(market_row.get("volume", 0) or 0)
    features["log_volume"] = math.log1p(vol)

    # Liquidity (log-transformed)
    liq = float(market_row.get("liquidity", 0) or 0)
    features["log_liquidity"] = math.log1p(liq)

    # Tick size
    features["tick_size"] = float(market_row.get("tick_size", 0.01) or 0.01)

    # NegRisk
    features["neg_risk"] = 1 if market_row.get("neg_risk") else 0

    return features


def compute_temporal_features(timestamp):
    """Compute time-of-day and day-of-week features.

    Uses cyclical encoding (sin/cos) so 23:59 is close to 00:00.

    Args:
        timestamp: unix timestamp

    Returns:
        dict of feature_name: value
    """
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    features = {}

    # Hour of day (cyclical)
    hour_frac = dt.hour + dt.minute / 60
    features["hour_sin"] = math.sin(2 * math.pi * hour_frac / 24)
    features["hour_cos"] = math.cos(2 * math.pi * hour_frac / 24)

    # Day of week (cyclical)
    dow = dt.weekday()
    features["dow_sin"] = math.sin(2 * math.pi * dow / 7)
    features["dow_cos"] = math.cos(2 * math.pi * dow / 7)

    # Is US market hours (9:30-16:00 ET = 14:30-21:00 UTC)
    features["us_market_hours"] = 1 if 14.5 <= hour_frac <= 21 else 0

    # Is pre-market catalyst window (8:00-9:00 ET = 13:00-14:00 UTC)
    # CPI, jobs, etc. release at 8:30 ET
    features["catalyst_window"] = 1 if 13 <= hour_frac <= 14 else 0

    return features


# ═══════════════════════════════════════════════════════
# LABEL COMPUTATION
# ═══════════════════════════════════════════════════════

def find_spike_peak(prices, trigger_idx, direction="UP"):
    """Find the peak of a spike after the trigger point.

    Args:
        prices: full price list
        trigger_idx: where the trigger fired
        direction: "UP" or "DOWN"

    Returns:
        (peak_idx, peak_price, is_genuine_spike) or None
    """
    t_trigger = prices[trigger_idx][0]
    p_trigger = prices[trigger_idx][1]

    # Find baseline (average price 5 min before trigger)
    baseline_prices = []
    for j in range(trigger_idx - 1, -1, -1):
        if prices[j][0] < t_trigger - 600:  # More than 10 min before
            break
        if prices[j][0] < t_trigger - 300:  # 5-10 min before
            baseline_prices.append(prices[j][1])

    if not baseline_prices:
        # Use earliest available
        baseline_prices = [prices[max(0, trigger_idx - 5)][1]]

    baseline = sum(baseline_prices) / len(baseline_prices) if baseline_prices else p_trigger

    # Search forward for peak within SPIKE_MAX_DURATION
    max_t = t_trigger + SPIKE_MAX_DURATION * 60
    peak_idx = trigger_idx
    peak_price = p_trigger

    for j in range(trigger_idx + 1, len(prices)):
        if prices[j][0] > max_t:
            break

        if direction == "UP" and prices[j][1] > peak_price:
            peak_price = prices[j][1]
            peak_idx = j
        elif direction == "DOWN" and prices[j][1] < peak_price:
            peak_price = prices[j][1]
            peak_idx = j

    # Check if this qualifies as a genuine spike
    if baseline <= 0:
        return peak_idx, peak_price, False

    total_move = abs(peak_price - baseline) / baseline * 100

    if total_move < SPIKE_MIN_MAGNITUDE:
        return peak_idx, peak_price, False

    # Check reversion after peak
    max_reversion = 0
    for j in range(peak_idx + 1, min(peak_idx + 120, len(prices))):
        if direction == "UP":
            reversion = (peak_price - prices[j][1]) / (peak_price - baseline) if peak_price > baseline else 0
        else:
            reversion = (prices[j][1] - peak_price) / (baseline - peak_price) if baseline > peak_price else 0

        max_reversion = max(max_reversion, reversion)

    is_spike = max_reversion >= SPIKE_MIN_REVERSION

    return peak_idx, peak_price, is_spike


def compute_peak_proximity(prices, current_idx, peak_idx, baseline, direction="UP"):
    """Compute peak proximity label (0.0 = just started, 1.0 = at peak).

    Uses the ACTUAL peak (known only in hindsight for training).
    """
    p_now = prices[current_idx][1]
    p_peak = prices[peak_idx][1]

    if direction == "UP":
        total_range = p_peak - baseline
        if total_range <= 0:
            return 0.5
        progress = (p_now - baseline) / total_range
    else:
        total_range = baseline - p_peak
        if total_range <= 0:
            return 0.5
        progress = (baseline - p_now) / total_range

    return max(0.0, min(1.0, progress))


# ═══════════════════════════════════════════════════════
# DATASET BUILDING
# ═══════════════════════════════════════════════════════

def find_trigger_points(prices, threshold=TRIGGER_THRESHOLD, window=TRIGGER_WINDOW):
    """Find all points where the scanner would have triggered.

    A trigger fires when price moves >threshold% within window minutes
    from any recent baseline.
    """
    triggers = []
    window_secs = window * 60
    last_trigger_t = 0

    for i in range(5, len(prices)):
        t_now, p_now = prices[i]

        # Don't trigger too close to previous trigger
        if t_now - last_trigger_t < 300:  # 5 min cooldown
            continue

        # Skip extreme prices
        if p_now < 0.05 or p_now > 0.95:
            continue

        # Check against prices in the lookback window
        for j in range(i - 1, -1, -1):
            t_j, p_j = prices[j]
            if t_now - t_j > window_secs:
                break
            if p_j <= 0:
                continue

            move = (p_now - p_j) / p_j * 100

            if abs(move) >= threshold:
                direction = "UP" if move > 0 else "DOWN"
                triggers.append({
                    "trigger_idx": i,
                    "baseline_idx": j,
                    "baseline_price": p_j,
                    "trigger_price": p_now,
                    "move_pct": move,
                    "direction": direction,
                    "timestamp": t_now,
                })
                last_trigger_t = t_now
                break

    return triggers


def build_training_data(args):
    """Build Stage 1 and Stage 2 training datasets."""
    threshold = getattr(args, "threshold", TRIGGER_THRESHOLD)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*90}")
    print(f"  FEATURE ENGINEERING — Building Training Data")
    print(f"  Trigger threshold: {threshold}%  |  Spike min: {SPIKE_MIN_MAGNITUDE}%")
    print(f"  Spike max duration: {SPIKE_MAX_DURATION}m  |  Min reversion: {SPIKE_MIN_REVERSION:.0%}")
    print(f"{'='*90}")

    # Get all markets with price data
    markets = conn.execute("""
        SELECT m.*, COALESCE(t.corrected_to, t.classified_as, m.event_type) as final_type
        FROM markets m
        LEFT JOIN taxonomy t ON m.slug = t.slug
        WHERE m.price_history_fetched = 1
    """).fetchall()

    print(f"\n  Markets with price data: {len(markets)}")

    stage1_rows = []
    stage2_rows = []
    stats = {
        "markets_processed": 0,
        "triggers_found": 0,
        "genuine_spikes": 0,
        "non_spikes": 0,
        "stage2_samples": 0,
        "by_type": {},
    }

    for mi, market in enumerate(markets):
        slug = market["slug"]

        # Get price history
        price_rows = conn.execute("""
            SELECT timestamp, price FROM price_history
            WHERE slug = ? ORDER BY timestamp ASC
        """, (slug,)).fetchall()

        if len(price_rows) < 10:
            continue

        prices = [(r["timestamp"], r["price"]) for r in price_rows]
        stats["markets_processed"] += 1

        # Get market metadata
        market_dict = dict(market)
        market_dict["event_type"] = market["final_type"] or "unknown"
        meta_features = compute_metadata_features(market_dict)

        # Find trigger points
        triggers = find_trigger_points(prices, threshold=threshold)

        for trigger in triggers:
            stats["triggers_found"] += 1
            tidx = trigger["trigger_idx"]
            direction = trigger["direction"]

            # Compute features at trigger time
            price_feats = compute_price_features(prices, tidx, slug)
            if price_feats is None:
                continue

            temp_feats = compute_temporal_features(trigger["timestamp"])

            # Find peak and determine if genuine spike (hindsight label)
            peak_idx, peak_price, is_spike = find_spike_peak(
                prices, tidx, direction)

            # ── Stage 1 sample ──
            stage1_row = {}
            stage1_row["slug"] = slug
            stage1_row["trigger_ts"] = trigger["timestamp"]
            stage1_row["direction"] = 1 if direction == "UP" else 0
            stage1_row["trigger_move_pct"] = trigger["move_pct"]

            # Price features
            stage1_row.update(price_feats)

            # Metadata features
            stage1_row.update(meta_features)

            # Temporal features
            stage1_row.update(temp_feats)

            # Label
            stage1_row["is_spike"] = 1 if is_spike else 0

            stage1_rows.append(stage1_row)

            etype = meta_features["event_type"]
            if etype not in stats["by_type"]:
                stats["by_type"][etype] = {"triggers": 0, "spikes": 0}
            stats["by_type"][etype]["triggers"] += 1

            if is_spike:
                stats["genuine_spikes"] += 1
                stats["by_type"][etype]["spikes"] += 1

                # ── Stage 2 samples (multiple per spike) ──
                # Sample at regular intervals from trigger to peak
                baseline = trigger["baseline_price"]

                for sidx in range(tidx, min(peak_idx + 1, len(prices))):
                    t_s = prices[sidx][0]

                    # Only sample every STAGE2_SAMPLE_INTERVAL seconds
                    if (t_s - prices[tidx][0]) % STAGE2_SAMPLE_INTERVAL > 30:
                        continue

                    s2_feats = compute_price_features(prices, sidx, slug)
                    if s2_feats is None:
                        continue

                    proximity = compute_peak_proximity(
                        prices, sidx, peak_idx, baseline, direction)

                    stage2_row = {}
                    stage2_row["slug"] = slug
                    stage2_row["sample_ts"] = t_s
                    stage2_row["direction"] = 1 if direction == "UP" else 0
                    stage2_row.update(s2_feats)
                    stage2_row.update(meta_features)
                    stage2_row.update(compute_temporal_features(t_s))

                    # Label
                    stage2_row["peak_proximity"] = proximity

                    stage2_rows.append(stage2_row)
                    stats["stage2_samples"] += 1
            else:
                stats["non_spikes"] += 1

        # Progress
        if (mi + 1) % 50 == 0:
            print(f"    {mi+1}/{len(markets)} markets | "
                  f"{stats['triggers_found']} triggers, "
                  f"{stats['genuine_spikes']} spikes, "
                  f"{stats['stage2_samples']} S2 samples")

    conn.close()

    # ── Write CSVs ──
    if stage1_rows:
        # Define column order (exclude non-numeric columns for training)
        exclude_cols = {"slug", "event_type", "timestamp", "trigger_ts",
                        "baseline_price", "price"}
        feature_cols = [k for k in stage1_rows[0].keys()
                        if k not in exclude_cols]

        s1_path = OUTPUT_DIR / "stage1_training.csv"
        with open(s1_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(stage1_rows[0].keys()))
            writer.writeheader()
            writer.writerows(stage1_rows)
        print(f"\n  Stage 1 written: {s1_path} ({len(stage1_rows)} rows)")

    if stage2_rows:
        s2_path = OUTPUT_DIR / "stage2_training.csv"
        with open(s2_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(stage2_rows[0].keys()))
            writer.writeheader()
            writer.writerows(stage2_rows)
        print(f"  Stage 2 written: {s2_path} ({len(stage2_rows)} rows)")

    # One-hot encode event types and write a separate mapping
    event_types = sorted(set(r.get("event_type", "unknown") for r in stage1_rows))
    type_map_path = OUTPUT_DIR / "event_type_map.json"
    with open(type_map_path, "w") as f:
        json.dump({t: i for i, t in enumerate(event_types)}, f, indent=2)
    print(f"  Event type map: {type_map_path} ({len(event_types)} types)")

    # ── Summary ──
    print(f"\n{'─'*90}")
    print(f"  DATASET SUMMARY")
    print(f"{'─'*90}")
    print(f"  Markets processed:       {stats['markets_processed']}")
    print(f"  Trigger points found:    {stats['triggers_found']}")
    print(f"  Genuine spikes (S1=1):   {stats['genuine_spikes']}")
    print(f"  Non-spikes (S1=0):       {stats['non_spikes']}")
    print(f"  Stage 2 samples:         {stats['stage2_samples']}")

    if stats["triggers_found"] > 0:
        spike_rate = stats["genuine_spikes"] / stats["triggers_found"]
        print(f"  Spike rate:              {spike_rate:.1%}")

    print(f"\n  By Event Type:")
    print(f"  {'Type':<15s} {'Triggers':>9s} {'Spikes':>7s} {'Rate':>6s}")
    print(f"  {'─'*40}")
    for etype, counts in sorted(stats["by_type"].items(), key=lambda x: -x[1]["triggers"]):
        rate = counts["spikes"] / counts["triggers"] if counts["triggers"] > 0 else 0
        print(f"  {etype:<15s} {counts['triggers']:>9d} {counts['spikes']:>7d} {rate:>5.0%}")

    if stage1_rows:
        # Feature statistics
        numeric_features = [k for k in stage1_rows[0].keys()
                            if k not in {"slug", "event_type", "timestamp",
                                         "trigger_ts", "baseline_price", "price"}
                            and isinstance(stage1_rows[0][k], (int, float))]

        print(f"\n  Feature Ranges (Stage 1):")
        print(f"  {'Feature':<25s} {'Min':>10s} {'Max':>10s} {'Mean':>10s}")
        print(f"  {'─'*58}")
        for feat in sorted(numeric_features)[:20]:
            vals = [r[feat] for r in stage1_rows if feat in r]
            if vals:
                print(f"  {feat:<25s} {min(vals):>10.3f} {max(vals):>10.3f} "
                      f"{sum(vals)/len(vals):>10.3f}")

    print(f"\n{'='*90}")


def cmd_stats(args):
    """Show existing training data statistics."""
    s1_path = OUTPUT_DIR / "stage1_training.csv"
    s2_path = OUTPUT_DIR / "stage2_training.csv"

    print(f"\n{'='*70}")
    print(f"  TRAINING DATA STATS")
    print(f"{'='*70}")

    for path, name in [(s1_path, "Stage 1"), (s2_path, "Stage 2")]:
        if path.exists():
            with open(path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            print(f"\n  {name}: {path}")
            print(f"    Rows: {len(rows)}")
            print(f"    Columns: {len(rows[0]) if rows else 0}")
            if rows and name == "Stage 1":
                spikes = sum(1 for r in rows if r.get("is_spike") == "1")
                print(f"    Spikes: {spikes} ({spikes/len(rows):.1%})")
                print(f"    Non-spikes: {len(rows) - spikes}")
            if rows and name == "Stage 2":
                proximities = [float(r["peak_proximity"]) for r in rows]
                print(f"    Proximity range: {min(proximities):.3f} — {max(proximities):.3f}")
                print(f"    Proximity mean: {sum(proximities)/len(proximities):.3f}")
        else:
            print(f"\n  {name}: not built yet")

    print(f"\n{'='*70}")


def cmd_inspect(args):
    """Inspect a specific market's price data and spikes."""
    slug = args.slug
    conn = sqlite3.connect(str(DB_PATH))

    prices = conn.execute("""
        SELECT timestamp, price FROM price_history
        WHERE slug = ? ORDER BY timestamp ASC
    """, (slug,)).fetchall()

    if not prices:
        print(f"  No price data for {slug}")
        return

    prices = [(r[0], r[1]) for r in prices]
    print(f"\n  Market: {slug}")
    print(f"  Price points: {len(prices)}")
    print(f"  Time range: {len(prices)} samples")
    print(f"  Price range: {min(p for _, p in prices):.3f} — {max(p for _, p in prices):.3f}")

    triggers = find_trigger_points(prices)
    print(f"  Trigger points: {len(triggers)}")

    for i, t in enumerate(triggers):
        peak_idx, peak_price, is_spike = find_spike_peak(
            prices, t["trigger_idx"], t["direction"])
        spike_str = "✅ SPIKE" if is_spike else "❌ not spike"
        print(f"\n  Trigger {i+1}: {t['direction']} {t['move_pct']:+.1f}% "
              f"@ {datetime.fromtimestamp(t['timestamp'], tz=timezone.utc).strftime('%H:%M')} "
              f"→ peak {peak_price:.3f} | {spike_str}")

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Spike Feature Engineering")
    subs = parser.add_subparsers(dest="command")

    p_build = subs.add_parser("build", help="Build training data")
    p_build.add_argument("--threshold", type=float, default=TRIGGER_THRESHOLD)

    subs.add_parser("stats", help="Show dataset stats")

    p_inspect = subs.add_parser("inspect", help="Inspect a market")
    p_inspect.add_argument("--slug", required=True)

    args = parser.parse_args()

    if args.command == "build":
        build_training_data(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "inspect":
        cmd_inspect(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
