"""
Leakage / causality unit tests for the info-bar pipeline (Cycle 58).

Deterministic synthetic bars (no DB). These tests are the executable form of the
protocol section 8 sign-off: if any feature, label, scaler, or split leaks
future information, a test here fails.

Run: pytest engines/infobar_lstm/tests/
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engines.infobar_lstm.pipeline import (
    BAR_COLUMNS, causal_features, triple_barrier_labels, label_uniqueness,
    walk_forward_folds, fit_scaler_train_only, frac_diff_ffd, ffd_weights,
    FEATURE_NAMES, FEATURE_WARMUP,
)
from engines.infobar_lstm.leakage_checks import (
    check_feature_causality, check_label_locality, check_scaler_train_only,
    check_walkforward_purge,
)


def make_synthetic_bars(n: int = 600, seed: int = 7,
                        drift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    open_ = np.empty(n)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    wiggle = np.abs(rng.normal(0, 0.004, n)) * close
    high = np.maximum(open_, close) + wiggle
    low = np.minimum(open_, close) - wiggle
    start = 1_777_000_000_000 + np.arange(n) * 60_000
    end = start + rng.integers(20_000, 60_000, n)
    qv = rng.uniform(5e5, 2e6, n)
    buy_frac = rng.uniform(0.3, 0.7, n)
    buy_q = qv * buy_frac
    sell_q = qv - buy_q
    return pd.DataFrame({
        "bar_index": np.arange(n),
        "start_timestamp": start,
        "end_timestamp": end,
        "open": open_, "high": high, "low": low, "close": close,
        "base_volume": qv / close,
        "quote_volume": qv,
        "tick_count": rng.integers(50, 500, n),
        "buy_quote": buy_q, "sell_quote": sell_q,
        "imbalance_quote": buy_q - sell_q,
    }, columns=BAR_COLUMNS)


def test_feature_causality():
    bars = make_synthetic_bars()
    ok, detail = check_feature_causality(bars)
    assert ok, detail


def test_label_locality():
    bars = make_synthetic_bars()
    ok, detail = check_label_locality(bars, H=20, vol_span=50)
    assert ok, detail


def test_scaler_train_only():
    bars = make_synthetic_bars()
    feats = causal_features(bars).fillna(0.0).to_numpy(dtype=float)
    ok, detail = check_scaler_train_only(feats, split=300)
    assert ok, detail


def test_walkforward_purge():
    bars = make_synthetic_bars()
    feats = causal_features(bars)
    lab = triple_barrier_labels(bars, H=20, vol_span=50)
    valid = feats.notna().all(axis=1).to_numpy().copy()
    valid[:FEATURE_WARMUP] = False
    usable = np.nonzero(valid & lab.labels.notna().to_numpy())[0]
    folds = walk_forward_folds(usable, n_folds=4, H=20, embargo=20)
    assert folds, "no folds produced"
    ok, detail = check_walkforward_purge(folds, H=20, embargo=20)
    assert ok, detail


def test_no_future_in_folds_explicit():
    """Belt-and-suspenders: assert directly that every train index + H is
    strictly before its fold's test_start."""
    bars = make_synthetic_bars()
    feats = causal_features(bars)
    lab = triple_barrier_labels(bars, H=15, vol_span=50)
    valid = feats.notna().all(axis=1).to_numpy().copy()
    valid[:FEATURE_WARMUP] = False
    usable = np.nonzero(valid & lab.labels.notna().to_numpy())[0]
    for fd in walk_forward_folds(usable, n_folds=3, H=15, embargo=15):
        assert fd.train_idx.max() + 15 < fd.test_start
        assert fd.train_idx.max() < fd.test_start - 15
        assert fd.test_idx.min() >= fd.test_start


def test_triple_barrier_directions():
    """A strictly-rising path labels mostly +1; a strictly-falling path -1."""
    up = make_synthetic_bars(n=300, seed=1, drift=0.02)
    dn = make_synthetic_bars(n=300, seed=1, drift=-0.02)
    lu = triple_barrier_labels(up, pt=1.0, sl=1.0, H=10, vol_span=20).labels
    ld = triple_barrier_labels(dn, pt=1.0, sl=1.0, H=10, vol_span=20).labels
    up_valid = lu.dropna()
    dn_valid = ld.dropna()
    assert (up_valid == 1).mean() > 0.7, (up_valid == 1).mean()
    assert (dn_valid == -1).mean() > 0.7, (dn_valid == -1).mean()


def test_labels_last_H_undefined():
    bars = make_synthetic_bars(n=200)
    H, vol_span = 20, 50
    lab = triple_barrier_labels(bars, H=H, vol_span=vol_span).labels
    # Last bar always undefined (no future bar at all).
    assert np.isnan(lab.iloc[-1])
    # Truncated-horizon tail must contain undefined labels (no fabricated 0s).
    assert lab.iloc[-H:].isna().any()
    # Interior (causal vol valid AND full horizon available) is fully labeled.
    assert lab.iloc[vol_span + 10: len(bars) - H - 1].notna().all()


def test_label_uniqueness_in_range():
    bars = make_synthetic_bars()
    lab = triple_barrier_labels(bars, H=20, vol_span=50)
    u = label_uniqueness(lab.touch_bar.dropna())
    assert 0.0 < u <= 1.0, u


def test_ffd_causal_and_warmup():
    w = ffd_weights(0.4)
    width = len(w)
    bars = make_synthetic_bars(n=width + 150)
    fd = frac_diff_ffd(np.log(bars["close"]), d=0.4)
    assert np.isnan(fd.iloc[: width - 1]).all()       # warmup NaNs
    assert fd.iloc[width:].notna().any()


def test_feature_names_present():
    bars = make_synthetic_bars()
    feats = causal_features(bars)
    assert list(feats.columns) == FEATURE_NAMES
    assert len(FEATURE_NAMES) == 15
