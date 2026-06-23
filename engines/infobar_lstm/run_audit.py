"""
engines/infobar_lstm/run_audit.py -- Run A: plumbing + leakage audit (NO FIT).

Exercises the full pipeline on real info bars -- load -> causal features ->
triple-barrier labels -> purged walk-forward -> train-only scaler -> sequences
-> cost/turnover -> benchmarks -- and runs the automated leakage checks. It
does NOT fit any model (the LSTM fit is gated to Run B; see model.py and
CYCLE58_VALIDATION_PROTOCOL.md).

The economic section uses the MOMENTUM/MA BASELINE as a stand-in strategy to
prove the cost + benchmark machinery computes end-to-end. That baseline is NOT
the model and its numbers are NOT an edge claim.

Usage (defaults = the pre-registered dry-run config, protocol section 5):
    python -m engines.infobar_lstm.run_audit
    python -m engines.infobar_lstm.run_audit --asset BTC --bar-type dollar \
        --threshold 1000000 --n-folds 4
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np

from .pipeline import (
    load_bars, causal_features, triple_barrier_labels, label_uniqueness,
    walk_forward_folds, fit_scaler_train_only, build_sequences,
    FEATURE_NAMES, FEATURE_WARMUP,
)
from .costs import evaluate_strategy, bars_per_year, RT_BPS_SWEEP
from .benchmarks import (
    buy_hold, momentum_signal, null_at_rate, trend_subperiods, subperiod_sharpe,
)
from .model import ModelSpec, param_count
from .leakage_checks import (
    check_feature_causality, check_label_locality, check_scaler_train_only,
    check_walkforward_purge,
)


def _iso(ts_ms) -> str:
    return datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc).strftime(
        "%Y-%m-%d")


def _hr(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def run_audit(asset: str, bar_type: str, threshold: float, n_folds: int,
              H: int, pt: float, sl: float, vol_span: int, seq_len: int) -> int:
    _hr(f"INFO-BAR LSTM -- RUN A (PLUMBING + LEAKAGE AUDIT, NO FIT)")
    print(f"  config: {asset} {bar_type} threshold={threshold:g}  "
          f"H={H} bars  pt/sl={pt}/{sl}  vol_span={vol_span}  seq_len={seq_len}")

    bars = load_bars(asset, bar_type, threshold)
    if bars.empty:
        print("  ERROR: no bars for this config.")
        return 1
    n = len(bars)
    print(f"  bars: {n}  span: {_iso(bars['start_timestamp'].iloc[0])} -> "
          f"{_iso(bars['end_timestamp'].iloc[-1])}")
    span_days = (float(bars['end_timestamp'].iloc[-1]
                       - bars['start_timestamp'].iloc[0]) / 86_400_000.0)
    bpy = bars_per_year(bars)
    print(f"  span_days: {span_days:.1f}  (~{span_days/7:.1f} weeks; ONE regime) "
          f" bars/year~{bpy:,.0f}")

    # ---- Features + labels ----
    _hr("FEATURES + LABELS")
    feats = causal_features(bars)
    lab = triple_barrier_labels(bars, pt=pt, sl=sl, H=H, vol_span=vol_span)
    labels = lab.labels

    feat_valid = feats.notna().all(axis=1).to_numpy().copy()
    feat_valid[:FEATURE_WARMUP] = False
    lab_valid = labels.notna().to_numpy()
    usable = np.nonzero(feat_valid & lab_valid)[0]
    print(f"  features: {len(FEATURE_NAMES)} -> {FEATURE_NAMES}")
    print(f"  usable bars (valid features AND label): {len(usable)} "
          f"({100*len(usable)/n:.1f}% of {n})")

    vc = labels.loc[usable].value_counts().to_dict()
    tot = sum(vc.values())
    dist = {int(k): f"{v} ({100*v/tot:.1f}%)" for k, v in sorted(vc.items())}
    print(f"  label dist {{-1,0,+1}}: {dist}")
    min_share = min(v / tot for v in vc.values()) if vc else 0
    print(f"  min class share: {min_share*100:.1f}%  "
          f"{'(DEGENERATE <10% -- per-class metrics only)' if min_share<0.1 else '(ok)'}")
    uniq = label_uniqueness(lab.touch_bar.loc[usable])
    print(f"  avg label uniqueness (AFML Ch.4): {uniq:.3f}  "
          f"-> effective N ~ {uniq*len(usable):,.0f} (raw {len(usable)})")

    spec = ModelSpec(input_size=len(FEATURE_NAMES), seq_len=seq_len)
    pc = param_count(spec)
    print(f"  model params (hidden={spec.hidden_size}, layers={spec.num_layers})"
          f": {pc:,}  -> examples:params raw {len(usable)/pc:.1f}:1, "
          f"effective {uniq*len(usable)/pc:.1f}:1")

    # ---- Walk-forward folds ----
    _hr("PURGED / EMBARGOED WALK-FORWARD")
    folds = walk_forward_folds(usable, n_folds=n_folds, H=H, embargo=H)
    for k, fd in enumerate(folds):
        print(f"  fold {k}: train={fd.train_idx.size:>6}  "
              f"test={fd.test_idx.size:>6}  "
              f"test_span {_iso(bars['end_timestamp'].iloc[fd.test_start])}->"
              f"{_iso(bars['end_timestamp'].iloc[fd.test_end])}")

    # ---- Scaler + sequences on one fold (shape proof) ----
    if folds:
        fd = folds[-1]
        Xtr_raw = feats.iloc[fd.train_idx].to_numpy(dtype=float)
        sc = fit_scaler_train_only(Xtr_raw)
        # Assemble a fold's sequences (train) to prove shapes; not fit.
        ftr = sc.transform(feats.iloc[fd.train_idx].to_numpy(dtype=float))
        ytr = labels.iloc[fd.train_idx].to_numpy(dtype=float)
        Xseq, yseq, _ = build_sequences(ftr, ytr, fd.train_idx, seq_len)
        print(f"  last fold: scaler fit on {Xtr_raw.shape[0]} train rows; "
              f"train sequences X={Xseq.shape}, y={yseq.shape}")

    # ---- Leakage self-checks ----
    _hr("LEAKAGE SELF-CHECKS (protocol section 8)")
    checks = [
        ("feature causality", check_feature_causality(bars)),
        ("label locality", check_label_locality(bars, H=H, vol_span=vol_span)),
        ("scaler train-only",
         check_scaler_train_only(feats.fillna(0.0).to_numpy(dtype=float),
                                 split=max(usable[len(usable)//2], 50))),
        ("walk-forward purge", check_walkforward_purge(folds, H=H, embargo=H)),
    ]
    all_pass = True
    for name, (ok, detail) in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        all_pass = all_pass and ok

    # ---- Economic machinery (MOMENTUM BASELINE, not the model) ----
    _hr("COST + BENCHMARK MACHINERY  (BASELINE = momentum/MA, NOT the model)")
    mom_window = 200
    mom = momentum_signal(bars, window=mom_window)
    up, down = trend_subperiods(bars)
    end_ts = bars["end_timestamp"].to_numpy()
    bh = buy_hold(bars)
    print(f"  Sharpe = DAILY-aggregated, annualized sqrt(365) (irregular event "
          f"clock). Baseline = {mom_window}-bar MA crossover.")
    print(f"  buy-hold: net_cum={bh.net_cum:+.4f}  Sharpe(own)={bh.sharpe_net:.2f}"
          f"  long_rate={bh.long_rate:.2f}")
    print(f"  {'RT_bps':>7} {'net_cum':>10} {'Sharpe':>8} {'turn/100':>9} "
          f"{'turn/day':>9} {'long%':>7}")
    for rt in RT_BPS_SWEEP:
        r = evaluate_strategy(bars, mom, rt_bps=rt)
        print(f"  {rt:>7.0f} {r.net_cum:>+10.4f} {r.sharpe_net:>8.2f} "
              f"{r.turnover_per_100:>9.2f} {r.turnover_per_day:>9.2f} "
              f"{r.long_rate:>6.1%}")
    # beta isolation + null-at-rate at the 40bps midpoint
    mid = evaluate_strategy(bars, mom, rt_bps=40.0)
    su = subperiod_sharpe(mid, up[:len(mid.net_ret)], end_ts[:len(mid.net_ret)])
    sd = subperiod_sharpe(mid, down[:len(mid.net_ret)], end_ts[:len(mid.net_ret)])
    print(f"  TC check (atlas Exp 1): gross_cum={mid.gross_cum:+.4f} vs "
          f"cost @40bps -> net_cum={mid.net_cum:+.4f} at {mid.turnover_per_day:.0f} "
          f"legs/day. Turnover x cost is the whole game.")
    print(f"  beta isolation @40bps: Sharpe up-bars={su:.2f}  down-bars={sd:.2f} "
          f"(vs buy-hold own {bh.sharpe_net:.2f})")
    nd = null_at_rate(bars, target_changes=mid.n_changes, rt_bps=40.0,
                      n_seeds=200)
    print(f"  null @40bps (turnover-matched, {mid.n_changes} changes): "
          f"mean={nd['mean']:.2f}  p05={nd['p05']:.2f}  p95={nd['p95']:.2f} "
          f"-> a real edge must clear p95={nd['p95']:.2f}")

    # ---- Hard pause ----
    _hr("HARD PAUSE")
    print("  Run A complete. NO MODEL WAS FIT (model.py:train_one_fold is gated;")
    print("  pass authorized_run_b=True only after Jeff reviews the protocol).")
    print("  Reminder: the numbers above are the momentum BASELINE exercising the")
    print("  machinery -- NOT an edge claim. Even a model result here would be ONE")
    print("  ~8-week regime: diagnostic, not durable (data-maturity gate, protocol")
    print("  section 7).")
    leak_status = "ALL PASS" if all_pass else "FAILURES PRESENT -- DO NOT PROCEED"
    print(f"  Leakage self-checks: {leak_status}")
    return 0 if all_pass else 2


def main() -> int:
    p = argparse.ArgumentParser(description="Info-bar LSTM Run A audit (no fit)")
    p.add_argument("--asset", default="BTC")
    p.add_argument("--bar-type", default="dollar")
    p.add_argument("--threshold", type=float, default=1_000_000.0)
    p.add_argument("--n-folds", type=int, default=4)
    p.add_argument("--H", type=int, default=20)
    p.add_argument("--pt", type=float, default=1.0)
    p.add_argument("--sl", type=float, default=1.0)
    p.add_argument("--vol-span", type=int, default=50)
    p.add_argument("--seq-len", type=int, default=32)
    a = p.parse_args()
    return run_audit(a.asset, a.bar_type, a.threshold, a.n_folds, a.H,
                     a.pt, a.sl, a.vol_span, a.seq_len)


if __name__ == "__main__":
    raise SystemExit(main())
