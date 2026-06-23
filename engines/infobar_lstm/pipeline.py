"""
engines/infobar_lstm/pipeline.py -- causal data pipeline for info-bar models.

Every public function here is CAUSAL: a value computed for bar i uses only
information available at bar i's close (bars <= i), and labels use only strictly
future bars (i+1 .. i+H). This is enforced by automated tests in
tests/test_leakage.py (prefix-invariance for features; window-locality for
labels; train-only scaler).

Leakage-surface contracts (protocol section 8):
  - features:  rolling/EWMA/diff windows are LEFT-CLOSED ending at i; no center,
               no negative shift, no full-sample statistics.
  - labels:    barrier volatility is a causal EWMA(<=i); target scans i+1..i+H.
  - scaler:    fit on train rows only; applied to test.
  - splits:    expanding walk-forward, PURGE train bars whose label window
               overlaps the test fold, plus an embargo gap.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DB = REPO / "data" / "crypto_data.db"

BAR_COLUMNS = [
    "bar_index", "start_timestamp", "end_timestamp",
    "open", "high", "low", "close",
    "base_volume", "quote_volume", "tick_count",
    "buy_quote", "sell_quote", "imbalance_quote",
]

# Rolling windows used by causal_features; the longest sets the warmup region.
_ROLL_SHORT = 5
_ROLL_MED = 10
_ROLL_LONG = 20
FEATURE_WARMUP = _ROLL_LONG  # bars to drop at the head (NaN rolling region)


# ---------------------------------------------------------------------------
# Bar loading
# ---------------------------------------------------------------------------

def load_bars(asset: str, bar_type: str, threshold_value: float,
              db_path: str | Path = DEFAULT_DB) -> pd.DataFrame:
    """Load one info-bar slice, ordered by end_timestamp (chronological).

    Returns a DataFrame indexed 0..N-1 (bar order) with BAR_COLUMNS. Read-only.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            f"SELECT {', '.join(BAR_COLUMNS)} FROM info_bars "
            "WHERE asset = ? AND bar_type = ? AND threshold_value = ? "
            "ORDER BY end_timestamp ASC, bar_index ASC",
            (asset, bar_type, float(threshold_value)),
        ).fetchall()
    finally:
        conn.close()
    df = pd.DataFrame(rows, columns=BAR_COLUMNS)
    return df


# ---------------------------------------------------------------------------
# Fractional differencing (AFML Ch. 5, fixed-width window) -- causal
# ---------------------------------------------------------------------------

def ffd_weights(d: float, thres: float = 1e-4) -> np.ndarray:
    """Fixed-width fractional-diff weights (oldest..newest)."""
    w = [1.0]
    k = 1
    while True:
        w_ = -w[-1] * (d - k + 1) / k
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1], dtype=float)


def frac_diff_ffd(series: pd.Series, d: float = 0.4,
                  thres: float = 1e-4) -> pd.Series:
    """Causal fractional differencing of a series.

    out[i] = w . series[i-width+1 : i+1]  (window ends at i -> causal).
    Stationary-with-memory transform of (log) price levels.
    """
    w = ffd_weights(d, thres)
    width = len(w)
    vals = series.to_numpy(dtype=float)
    out = np.full(len(series), np.nan)
    if width <= len(vals):
        # Vectorized causal correlation: out[i] = w . vals[i-width+1 : i+1].
        # Windows containing a NaN propagate NaN (matmul), preserving warmup.
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(vals, width)      # (n-width+1, width)
        out[width - 1:] = windows @ w
    return pd.Series(out, index=series.index, name=f"ffd_{d}")


# ---------------------------------------------------------------------------
# Causal features
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "log_ret_1", "log_ret_2", "log_ret_3", "log_ret_5", "log_ret_10",
    "imb_ratio", "buy_frac", "imb_ma_5", "imb_ma_10",
    "dur_ratio", "vol_ratio", "tick_ratio", "range_frac", "rv_20", "ffd_logclose",
]


def causal_features(bars: pd.DataFrame, ffd_d: float = 0.4) -> pd.DataFrame:
    """Compute the causal feature matrix (one row per bar).

    All columns use only bars <= i. Verified by prefix-invariance test:
    features computed on bars[:i+1] match features computed on the full frame
    at row i. The head FEATURE_WARMUP rows carry NaNs (rolling warmup) and are
    dropped downstream.
    """
    b = bars
    close = b["close"].astype(float)
    logc = np.log(close)
    f = pd.DataFrame(index=b.index)

    # Stationary returns (close_i / close_{i-k}); causal diffs.
    for k in (1, 2, 3, 5, 10):
        f[f"log_ret_{k}"] = logc.diff(k)

    # Per-bar signed order flow (bar i's own closed values -> causal).
    qv = b["quote_volume"].astype(float).replace(0.0, np.nan)
    f["imb_ratio"] = (b["imbalance_quote"].astype(float) / qv).clip(-1, 1)
    bq = b["buy_quote"].astype(float)
    sq = b["sell_quote"].astype(float)
    f["buy_frac"] = bq / (bq + sq).replace(0.0, np.nan)

    # Rolling means of signed flow (left-closed, causal).
    f["imb_ma_5"] = f["imb_ratio"].rolling(_ROLL_SHORT).mean()
    f["imb_ma_10"] = f["imb_ratio"].rolling(_ROLL_MED).mean()

    # Activity / intensity ratios vs causal rolling mean.
    dur = (b["end_timestamp"] - b["start_timestamp"]).astype(float).clip(lower=1)
    f["dur_ratio"] = dur / dur.rolling(_ROLL_LONG).mean()
    f["vol_ratio"] = qv / qv.rolling(_ROLL_LONG).mean()
    tc = b["tick_count"].astype(float)
    f["tick_ratio"] = tc / tc.rolling(_ROLL_LONG).mean()

    # Intrabar range and realized vol (causal).
    f["range_frac"] = (b["high"].astype(float) - b["low"].astype(float)) / close
    f["rv_20"] = f["log_ret_1"].rolling(_ROLL_LONG).std()

    # Stationary-with-memory price level (causal FFD of log close).
    f["ffd_logclose"] = frac_diff_ffd(logc, d=ffd_d).values

    return f[FEATURE_NAMES]


# ---------------------------------------------------------------------------
# Triple-barrier labels (event-time, in bar space) -- causal vol, future target
# ---------------------------------------------------------------------------

def ewma_vol(bars: pd.DataFrame, span: int = 50) -> pd.Series:
    """Causal EWMA volatility of bar log-returns (uses bars <= i)."""
    logret = np.log(bars["close"].astype(float)).diff()
    return logret.ewm(span=span, min_periods=span).std()


@dataclass
class LabelResult:
    labels: pd.Series          # {-1, 0, +1}, NaN where undefined (last H bars)
    touch_bar: pd.Series       # bar index where a barrier was first hit (or i+H)
    ret_at_touch: pd.Series    # realized log-return at the touch
    sigma: pd.Series           # causal vol used to size the barriers
    pt: float
    sl: float
    H: int


def triple_barrier_labels(bars: pd.DataFrame, pt: float = 1.0, sl: float = 1.0,
                          H: int = 20, vol_span: int = 50) -> LabelResult:
    """Triple-barrier labels in BAR SPACE.

    For each bar i (entry at i's close):
      upper = close_i * exp(+pt * sigma_i),  lower = close_i * exp(-sl * sigma_i)
      scan bars i+1 .. i+H using each bar's high/low; first touch wins.
      label = +1 (upper first) / -1 (lower first) / 0 (vertical, no touch).
    sigma_i is a causal EWMA(<=i). Labels for the final H bars are NaN.

    Causality: label_i depends only on sigma_i (bars <= i) and bars i+1..i+H.
    """
    n = len(bars)
    close = bars["close"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    sigma = ewma_vol(bars, span=vol_span).to_numpy(dtype=float)

    labels = np.full(n, np.nan)
    touch = np.full(n, np.nan)
    ret_touch = np.full(n, np.nan)

    for i in range(n):
        s = sigma[i]
        if not np.isfinite(s) or s <= 0 or i + 1 >= n:
            continue
        full_horizon = (i + H) <= (n - 1)           # all H future bars exist
        j_end = min(i + H, n - 1)
        if j_end <= i:
            continue
        upper = close[i] * np.exp(pt * s)
        lower = close[i] * np.exp(-sl * s)
        hi = high[i + 1: j_end + 1]
        lo = low[i + 1: j_end + 1]
        up_hit = np.nonzero(hi >= upper)[0]
        dn_hit = np.nonzero(lo <= lower)[0]
        up_k = up_hit[0] if up_hit.size else None
        dn_k = dn_hit[0] if dn_hit.size else None

        if up_k is None and dn_k is None:
            # No barrier touched within the available bars. If the full horizon
            # was not available (ran out of data), the outcome is UNKNOWN ->
            # leave NaN rather than fabricate a 0 from a truncated window.
            if not full_horizon:
                continue
            lab, k = 0, (j_end - (i + 1))           # vertical barrier
        elif dn_k is None or (up_k is not None and up_k <= dn_k):
            lab, k = 1, up_k
        else:
            lab, k = -1, dn_k
        j = i + 1 + k
        labels[i] = lab
        touch[i] = j
        ret_touch[i] = np.log(close[j] / close[i])

    idx = bars.index
    return LabelResult(
        labels=pd.Series(labels, index=idx, name="label"),
        touch_bar=pd.Series(touch, index=idx, name="touch_bar"),
        ret_at_touch=pd.Series(ret_touch, index=idx, name="ret_at_touch"),
        sigma=pd.Series(sigma, index=idx, name="sigma"),
        pt=pt, sl=sl, H=H,
    )


def label_uniqueness(touch_bar: pd.Series) -> float:
    """Average label uniqueness (AFML Ch. 4): mean of 1/concurrency.

    Concurrency at bar t = number of labels whose [i, touch_i] span covers t.
    Low uniqueness => effective sample << raw bar count. O(N) via a difference
    array over the label spans.
    """
    valid = touch_bar.dropna()
    if valid.empty:
        return float("nan")
    starts = valid.index.to_numpy(dtype=int)
    ends = valid.to_numpy(dtype=int)
    # Size by the largest bar index touched, not the (possibly subset) length.
    n = int(max(starts.max(), ends.max())) + 2
    diff = np.zeros(n + 1)
    for i, j in zip(starts, ends):
        diff[i] += 1
        diff[min(j, n - 1) + 1] -= 1
    conc = np.cumsum(diff[:n])
    uniq = []
    for i, j in zip(starts, ends):
        span = conc[i: min(j, n - 1) + 1]
        span = span[span > 0]
        if span.size:
            uniq.append(float(np.mean(1.0 / span)))
    return float(np.mean(uniq)) if uniq else float("nan")


# ---------------------------------------------------------------------------
# Walk-forward folds (expanding, purged, embargoed)
# ---------------------------------------------------------------------------

@dataclass
class Fold:
    train_idx: np.ndarray
    test_idx: np.ndarray
    test_start: int
    test_end: int


def walk_forward_folds(usable_idx: np.ndarray, n_folds: int, H: int,
                       embargo: Optional[int] = None) -> list[Fold]:
    """Expanding-window walk-forward with purge + embargo.

    usable_idx: sorted array of bar indices that have BOTH valid features and
                valid labels (NaN rows already excluded).
    Splits usable_idx into n_folds+1 contiguous chunks; fold k trains on chunks
    0..k and tests on chunk k+1.

    PURGE: drop any train bar i whose label window [i, i+H] reaches into the
    test fold (overlapping-label leakage).
    EMBARGO: drop the `embargo` train bars immediately preceding test_start.
    """
    if embargo is None:
        embargo = H
    u = np.asarray(usable_idx)
    n = len(u)
    if n_folds < 1 or n < (n_folds + 1) * 2:
        raise ValueError(f"not enough usable bars ({n}) for {n_folds} folds")
    bounds = np.linspace(0, n, n_folds + 2, dtype=int)
    folds: list[Fold] = []
    for k in range(1, n_folds + 1):
        tr_pos = np.arange(0, bounds[k])
        te_pos = np.arange(bounds[k], bounds[k + 1])
        if te_pos.size == 0:
            continue
        train = u[tr_pos]
        test = u[te_pos]
        test_start = int(test[0])
        test_end = int(test[-1])
        # PURGE: train bar i leaks if its label window i..i+H touches test_start.
        train = train[train + H < test_start]
        # EMBARGO: extra gap before test_start.
        train = train[train < test_start - embargo]
        if train.size == 0:
            continue
        folds.append(Fold(train_idx=train, test_idx=test,
                          test_start=test_start, test_end=test_end))
    return folds


# ---------------------------------------------------------------------------
# Train-only scaler
# ---------------------------------------------------------------------------

@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


def fit_scaler_train_only(X_train: np.ndarray) -> Scaler:
    """Fit standardization stats on TRAIN rows only (leakage control)."""
    mean = np.nanmean(X_train, axis=0)
    std = np.nanstd(X_train, axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return Scaler(mean=mean, std=std)


# ---------------------------------------------------------------------------
# Sequence assembly (for the LSTM; assembled but not fit this cycle)
# ---------------------------------------------------------------------------

def build_sequences(features: np.ndarray, labels: np.ndarray,
                    anchor_idx: np.ndarray, seq_len: int) -> tuple:
    """Assemble (X, y, anchors) where X[k] = features[i-seq_len+1 .. i].

    `features`/`labels` are aligned, contiguous arrays over a fold's index set;
    `anchor_idx` is the original bar index of each row (to detect gaps from
    purging). A sequence is emitted only when the seq_len lookback rows are
    contiguous in the original bar order (no purge gap inside the window).
    """
    X, y, anc = [], [], []
    for k in range(seq_len - 1, len(features)):
        win = anchor_idx[k - seq_len + 1: k + 1]
        if win[-1] - win[0] != seq_len - 1:       # gap inside lookback -> skip
            continue
        if not np.isfinite(labels[k]):
            continue
        X.append(features[k - seq_len + 1: k + 1])
        y.append(labels[k])
        anc.append(anchor_idx[k])
    if not X:
        return (np.empty((0, seq_len, features.shape[1])),
                np.empty((0,)), np.empty((0,)))
    return np.asarray(X, dtype=np.float32), np.asarray(y), np.asarray(anc)
