#!/usr/bin/env python3
"""
scripts/funding_monitor.py
===========================
Live funding rate carry monitor for the Praxis trading platform.

Runs at each Binance funding window (00:00, 08:00, 16:00 UTC) and
produces a signal report: which assets pass the P > 0.70 gate,
which config the RF selected, and the expected carry for the hold period.

Usage:
    # Single run (one-shot report)
    python scripts/funding_monitor.py

    # Loop mode (runs at each 8h funding window)
    python scripts/funding_monitor.py --loop

    # Custom gate threshold
    python scripts/funding_monitor.py --gate 0.80

    # Use specific model file
    python scripts/funding_monitor.py --models output/funding_rate/cpo/phase3_models.joblib

    # Save report to file
    python scripts/funding_monitor.py --output reports/funding_$(date +%Y%m%d_%H%M).txt
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_ASSETS   = ["BTC", "ETH", "SOL", "XRP", "ADA", "AVAX"]  # BNB excluded
DEFAULT_MODELS   = "output/funding_rate/cpo/phase3_models_funding.joblib"
DEFAULT_GATE     = 0.70
DEFAULT_CACHE    = "data/funding_cache"
QUOTE            = "USDT"
LOOKBACK_DAYS    = 35   # days of history needed for features
FUNDING_WINDOWS  = [0, 8, 16]  # UTC hours of Binance funding payments


# ── Data fetching ──────────────────────────────────────────────────────────────

def fetch_live_data(assets: list[str], lookback_days: int = LOOKBACK_DAYS,
                    cache_dir: str = DEFAULT_CACHE) -> dict:
    """
    Fetch spot bars, perp bars, and funding rates for the past lookback_days.
    Uses the same CCXT / Binance source as training.
    """
    import ccxt

    now       = datetime.now(timezone.utc)
    start_str = (now - timedelta(days=lookback_days + 2)).strftime("%Y-%m-%d")
    end_str   = (now + timedelta(days=1)).strftime("%Y-%m-%d")  # buffer for today

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    spot_exchange = ccxt.binance({"enableRateLimit": True})
    perp_exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })

    spot_data    = {}
    perp_data    = {}

    def fetch_ohlcv(exchange, symbol, timeframe="1h", since_ms=None, end_ms=None):
        all_bars, cursor = [], since_ms
        while cursor < end_ms:
            bars = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=1000)
            if not bars:
                break
            bars = [b for b in bars if b[0] < end_ms]
            if not bars:
                break
            all_bars.extend(bars)
            last = bars[-1][0]
            if last <= cursor:
                break
            cursor = last + 1
        return all_bars

    since_ms = int((now - timedelta(days=lookback_days + 2)).timestamp() * 1000)
    end_ms   = int(now.timestamp() * 1000)

    for asset in assets:
        spot_symbol = f"{asset}/{QUOTE}"
        perp_symbol = f"{asset}/{QUOTE}:{QUOTE}"

        # ── Spot ──
        print(f"  Fetching {asset} spot...", end=" ", flush=True)
        try:
            bars = fetch_ohlcv(spot_exchange, spot_symbol, since_ms=since_ms, end_ms=end_ms)
            if bars:
                df = pd.DataFrame(bars, columns=["timestamp","open","high","low","close","volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()
                spot_data[asset] = df
                print(f"{len(df)} bars ✓")
            else:
                print("no data")
        except Exception as e:
            print(f"FAIL: {e}")

        # ── Perp ──
        print(f"  Fetching {asset} perp...", end=" ", flush=True)
        try:
            bars = fetch_ohlcv(perp_exchange, perp_symbol, since_ms=since_ms, end_ms=end_ms)
            if bars:
                df = pd.DataFrame(bars, columns=["timestamp","open","high","low","close","volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()
                perp_data[asset] = {"perp": df}
                print(f"{len(df)} bars ✓")
            else:
                print("no data")
        except Exception as e:
            print(f"FAIL: {e}")

        # ── Funding rates ──
        if asset not in perp_data:
            continue
        print(f"  Fetching {asset} funding...", end=" ", flush=True)
        try:
            all_fr = []
            cursor = since_ms
            while cursor < end_ms:
                recs = perp_exchange.fetch_funding_rate_history(
                    perp_symbol, since=cursor, limit=500
                )
                if not recs:
                    break
                recs = [r for r in recs if r["timestamp"] < end_ms]
                if not recs:
                    break
                all_fr.extend(recs)
                last = recs[-1]["timestamp"]
                if last <= cursor:
                    break
                cursor = last + 1
            if all_fr:
                fr_df = pd.DataFrame([
                    {"timestamp": pd.to_datetime(r["timestamp"], unit="ms", utc=True),
                     "rate": float(r["fundingRate"])}
                    for r in all_fr
                ]).drop_duplicates("timestamp").set_index("timestamp").sort_index()
                perp_data[asset]["funding"] = fr_df["rate"]
                print(f"{len(fr_df)} payments ✓")
            else:
                print("no data")
                del perp_data[asset]
        except Exception as e:
            print(f"FAIL: {e}")
            if asset in perp_data:
                del perp_data[asset]

    return {"spot": spot_data, "perp": perp_data}


# ── Feature computation ────────────────────────────────────────────────────────

def compute_live_features(data: dict, assets: list[str]) -> dict[str, np.ndarray | None]:
    """Compute the 11 funding features for each asset as of now."""
    from engines.funding_rate_strategy import _compute_funding_features

    now     = datetime.now(timezone.utc)
    now_str = now.strftime("%Y-%m-%d %H:%M:%S+00:00")

    features = {}
    for asset in assets:
        spot      = data["spot"].get(asset)
        perp_data = data["perp"].get(asset)
        if spot is None or perp_data is None:
            features[asset] = None
            continue

        feat = _compute_funding_features(
            spot, perp_data["perp"], perp_data["funding"], now_str
        )
        features[asset] = feat

    return features


# ── Inference ──────────────────────────────────────────────────────────────────

def run_inference(
    features:    dict[str, np.ndarray | None],
    models_path: str,
    assets:      list[str],
    gate:        float = DEFAULT_GATE,
) -> list[dict]:
    """
    Load trained RF models and run inference for each asset.
    Returns list of signal dicts, sorted by P(profitable) descending.
    """
    import joblib
    from engines.funding_rate_strategy import generate_funding_param_grid
    from engines.cpo_core import predict_model

    print(f"\n  Loading models from {models_path}...")
    models = joblib.load(models_path)
    configs = generate_funding_param_grid()

    signals = []
    missing_models = []
    for asset in assets:
        model_id = f"{asset}_FUNDING"
        tm       = models.get(model_id, {})
        feat     = features.get(asset)

        if tm.get("model") is None:
            missing_models.append(asset)
            continue
        if feat is None:
            continue

        best_config, p_prob, exp_ret = predict_model(tm, feat, configs)

        # Current funding rate for display
        # (re-extracted from features — feature[1] = ann_rate)
        ann_rate = float(feat[1]) if feat is not None and len(feat) > 1 else 0.0
        basis    = float(feat[6]) if feat is not None and len(feat) > 6 else 0.0
        pct_pos  = float(feat[4]) if feat is not None and len(feat) > 4 else 0.0

        signals.append({
            "asset":          asset,
            "model_id":       model_id,
            "p_profitable":   p_prob,
            "exp_return":     exp_ret,
            "above_gate":     p_prob > gate,
            "hold_days":      best_config.hold_days,
            "min_ann_pct":    best_config.min_funding_ann_pct,
            "ann_rate":       ann_rate,
            "basis_pct":      basis,
            "pct_positive":   pct_pos,
            "base_rate":      tm.get("base_rate", 0.0),
        })

    if missing_models:
        print(f"  ⚠ No trained models for: {', '.join(missing_models)}")
        print(f"    Retrain: python scripts/run_cpo.py --strategy funding_rate "
              f"--feature-mode funding --assets {','.join(missing_models)} "
              f"--training-start 2024-01-01 --training-end 2024-12-31 "
              f"--cache-dir data/funding_cache phase3")

    return sorted(signals, key=lambda x: x["p_profitable"], reverse=True)


# ── Report formatting ─────────────────────────────────────────────────────────

def format_report(signals: list[dict], gate: float, as_of: datetime) -> str:
    """Format a human-readable signal report."""
    lines = []
    lines.append("=" * 70)
    lines.append("FUNDING RATE CARRY MONITOR")
    lines.append(f"As of: {as_of.strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"Gate:  P > {gate:.2f}")
    lines.append("=" * 70)

    # Active signals (above gate)
    active = [s for s in signals if s["above_gate"]]
    lines.append(f"\n{'▶ ACTIVE SIGNALS':} ({len(active)} assets)\n")

    if active:
        lines.append(f"  {'Asset':<8} {'P(profit)':>10} {'Ann Rate':>10} {'Basis%':>8} "
                     f"{'Pct+':>7} {'Hold':>6} {'ExpRet':>8}")
        lines.append("  " + "-" * 62)
        for s in active:
            lines.append(
                f"  {s['asset']:<8} {s['p_profitable']:>10.3f} "
                f"{s['ann_rate']:>9.1f}% {s['basis_pct']:>7.3f}% "
                f"{s['pct_positive']:>6.0%} {s['hold_days']:>4}d "
                f"{s['exp_return']:>+8.4f}"
            )
    else:
        lines.append("  No assets above gate — staying flat.")

    # Monitoring (below gate)
    inactive = [s for s in signals if not s["above_gate"]]
    if inactive:
        lines.append(f"\n{'◎ MONITORING':} ({len(inactive)} assets — below gate)\n")
        lines.append(f"  {'Asset':<8} {'P(profit)':>10} {'Ann Rate':>10} {'Basis%':>8} {'Pct+':>7}")
        lines.append("  " + "-" * 50)
        for s in inactive:
            lines.append(
                f"  {s['asset']:<8} {s['p_profitable']:>10.3f} "
                f"{s['ann_rate']:>9.1f}% {s['basis_pct']:>7.3f}% "
                f"{s['pct_positive']:>6.0%}"
            )

    # Summary
    lines.append("\n" + "─" * 70)
    if active:
        max_p = max(s["p_profitable"] for s in active)
        lines.append(f"  ACTION: Enter carry on {[s['asset'] for s in active]} "
                     f"(highest P = {max_p:.3f})")
        lines.append(f"  HOLD:   {active[0]['hold_days']} days (RF-selected)")
        lines.append(f"  SETUP:  Long spot + Short perp, delta-neutral")
        lines.append(f"  TC:     ~4 bps per leg (Binance maker)")
    else:
        lines.append("  ACTION: No trade — funding conditions unfavorable.")
    lines.append("=" * 70)

    return "\n".join(lines)


# ── Next window calculation ────────────────────────────────────────────────────

def next_funding_window(now: datetime) -> datetime:
    """Calculate the next Binance funding payment window (00/08/16 UTC)."""
    today = now.replace(minute=0, second=0, microsecond=0)
    for h in FUNDING_WINDOWS:
        candidate = today.replace(hour=h)
        if candidate > now:
            return candidate
    # Next day 00:00
    return (today + timedelta(days=1)).replace(hour=0)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_once(args) -> str:
    """Run one monitoring cycle and return the report string."""
    now    = datetime.now(timezone.utc)
    assets = args.assets.split(",") if args.assets else DEFAULT_ASSETS

    print(f"\n[{now.strftime('%H:%M UTC')}] Fetching live data for {assets}...")
    data = fetch_live_data(assets, cache_dir=args.cache_dir)

    print("\n  Computing features...")
    features = compute_live_features(data, assets)
    n_ok = sum(1 for v in features.values() if v is not None)
    print(f"  Features computed: {n_ok}/{len(assets)} assets")

    print(f"\n  Running RF inference (gate P > {args.gate})...")
    signals = run_inference(features, args.models, assets, gate=args.gate)

    report = format_report(signals, args.gate, now)
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Live funding rate carry monitor"
    )
    parser.add_argument("--assets", type=str, default=None,
                        help="Comma-separated assets (default: BTC,ETH,SOL,XRP,ADA,AVAX)")
    parser.add_argument("--models", type=str, default=DEFAULT_MODELS,
                        help=f"Path to trained models .joblib (default: {DEFAULT_MODELS})")
    parser.add_argument("--gate", type=float, default=DEFAULT_GATE,
                        help=f"P(profitable) gate threshold (default: {DEFAULT_GATE})")
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE,
                        help="Cache directory for market data")
    parser.add_argument("--loop", action="store_true",
                        help="Run at each 8h funding window (00:00, 08:00, 16:00 UTC)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save report to file (in addition to stdout)")
    parser.add_argument("--webhook", type=str, default=None,
                        help="POST report to webhook URL when signals are active")
    args = parser.parse_args()

    if not Path(args.models).exists():
        print(f"ERROR: Models file not found: {args.models}")
        print("Run phase3 first:")
        print("  python scripts/run_cpo.py --strategy funding_rate ... phase3")
        sys.exit(1)

    if not args.loop:
        # One-shot
        report = run_once(args)
        print("\n" + report)

        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                f.write(report)
            print(f"\nReport saved to {args.output}")

        if args.webhook:
            _post_webhook(args.webhook, report)

    else:
        # Loop mode: run at each funding window
        print(f"Loop mode: running at 00:00, 08:00, 16:00 UTC")
        print("Press Ctrl+C to stop.\n")

        while True:
            try:
                report = run_once(args)
                print("\n" + report)

                if args.output:
                    ts  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
                    out = args.output.replace("{ts}", ts)
                    Path(out).parent.mkdir(parents=True, exist_ok=True)
                    with open(out, "w") as f:
                        f.write(report)

                if args.webhook:
                    # Only post when signals are active
                    if "ACTION: Enter" in report:
                        _post_webhook(args.webhook, report)

                # Sleep until next window
                now  = datetime.now(timezone.utc)
                next_w = next_funding_window(now)
                sleep_s = (next_w - now).total_seconds() + 60  # +60s buffer
                print(f"\n  Next run: {next_w.strftime('%Y-%m-%d %H:%M UTC')} "
                      f"(in {sleep_s/3600:.1f}h)")
                time.sleep(sleep_s)

            except KeyboardInterrupt:
                print("\nMonitor stopped.")
                break
            except Exception as e:
                print(f"\nERROR: {e}")
                print("Retrying in 5 minutes...")
                time.sleep(300)


def _post_webhook(url: str, report: str):
    """POST report to a webhook URL (Slack, Discord, custom)."""
    import urllib.request, json
    try:
        payload = json.dumps({"text": f"```\n{report}\n```"}).encode()
        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
        print("  Webhook posted ✓")
    except Exception as e:
        print(f"  Webhook failed: {e}")


if __name__ == "__main__":
    main()
