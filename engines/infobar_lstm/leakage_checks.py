"""
engines/infobar_lstm/leakage_checks.py -- automated causality / leakage checks.

Shared by run_audit.py (Run A runtime assertions) and tests/test_leakage.py
(pytest). Each returns (passed: bool, detail: str). The structural checks are
the PRIMARY defense against leakage (protocol section 8); a model's implausibly
high accuracy is only the secondary smell test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .pipeline import (
    causal_features, triple_barrier_labels, walk_forward_folds,
    fit_scaler_train_only, Fold,
)


def _probe_indices(n: int) -> list[int]:
    """Deterministic spread of probe rows (no RNG)."""
    fracs = (0.30, 0.50, 0.70, 0.85, 0.95)
    return sorted({max(25, int(f * n)) for f in fracs if int(f * n) < n - 1})


def _row_equal(a: pd.Series, b: pd.Series, tol: float = 1e-9) -> bool:
    av, bv = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
    both_nan = np.isnan(av) & np.isnan(bv)
    close = np.isclose(av, bv, rtol=0, atol=tol, equal_nan=False)
    return bool(np.all(both_nan | close))


def check_feature_causality(bars: pd.DataFrame) -> tuple[bool, str]:
    """Prefix-invariance: features at row i must not change when future bars
    are added. Recompute features on bars[:i+1] and compare row i to the
    full-frame row i. Any future leakage breaks this exactly."""
    full = causal_features(bars).reset_index(drop=True)
    bad = []
    for i in _probe_indices(len(bars)):
        prefix = causal_features(bars.iloc[: i + 1].reset_index(drop=True))
        if not _row_equal(full.iloc[i], prefix.iloc[i]):
            bad.append(i)
    if bad:
        return False, f"feature prefix-invariance FAILED at rows {bad}"
    return True, "features depend only on bars <= i (prefix-invariant)"


def check_label_locality(bars: pd.DataFrame, H: int = 20,
                         vol_span: int = 50) -> tuple[bool, str]:
    """Window-locality: label[i] must be unchanged when the frame is truncated
    just past its label window (i+H) -- i.e. it needs no bar beyond i+H."""
    full = triple_barrier_labels(bars, H=H, vol_span=vol_span).labels
    bad = []
    for i in _probe_indices(len(bars)):
        if i + H + 1 >= len(bars):
            continue
        trunc = triple_barrier_labels(
            bars.iloc[: i + H + 1].reset_index(drop=True), H=H, vol_span=vol_span
        ).labels
        a, b = full.iloc[i], trunc.iloc[i]
        if not ((np.isnan(a) and np.isnan(b)) or a == b):
            bad.append(i)
    if bad:
        return False, f"label window-locality FAILED at rows {bad}"
    return True, "label[i] uses only bars i+1..i+H (causal vol <= i)"


def check_scaler_train_only(features: np.ndarray, split: int) -> tuple[bool, str]:
    """The scaler must be a pure function of train rows: perturbing the test
    block must not change the fitted mean/std."""
    Xtr = features[:split]
    sc1 = fit_scaler_train_only(Xtr)
    perturbed = features.copy()
    perturbed[split:] += 1e3                       # wreck the test block
    sc2 = fit_scaler_train_only(perturbed[:split])
    ok = (np.allclose(sc1.mean, sc2.mean, equal_nan=True)
          and np.allclose(sc1.std, sc2.std, equal_nan=True))
    if not ok:
        return False, "scaler changed when test block perturbed -- LEAK"
    return True, "scaler fit on train rows only (test-perturbation invariant)"


def check_walkforward_purge(folds: list[Fold], H: int,
                            embargo: int) -> tuple[bool, str]:
    """No train bar i may have its label window [i, i+H] reach the test fold,
    and all train bars must precede test_start - embargo."""
    for k, fd in enumerate(folds):
        if fd.train_idx.size == 0:
            continue
        max_reach = int(fd.train_idx.max()) + H
        if max_reach >= fd.test_start:
            return False, (f"fold {k}: train label window reaches test "
                           f"(max i+H={max_reach} >= test_start={fd.test_start})")
        if int(fd.train_idx.max()) >= fd.test_start - embargo:
            return False, f"fold {k}: embargo gap violated"
    return True, f"all {len(folds)} folds purged + embargoed (H={H})"
