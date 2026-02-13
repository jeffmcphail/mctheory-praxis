"""
ML Momentum (Phase 4.6, §14.4).

Implements Advances in Financial Machine Learning (AFML) techniques:
- Triple barrier labeling (de Prado ch. 3)
- Combinatorial purged cross-validation (de Prado ch. 7)
- Sample uniqueness / average uniqueness weighting (de Prado ch. 4)
- Feature engineering (returns, volatility, momentum features)

Usage:
    labels = triple_barrier_label(prices, pt=0.02, sl=0.01, max_bars=20)
    cv = PurgedKFold(n_splits=5, embargo_pct=0.01)
    for train, test in cv.split(X, labels):
        model.fit(X[train], labels.side[train])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


# ═══════════════════════════════════════════════════════════════════
#  Triple Barrier Labeling
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TripleBarrierEvent:
    """One labeling event from triple barrier method."""
    entry_idx: int
    exit_idx: int
    entry_price: float
    exit_price: float
    return_pct: float
    barrier_hit: str          # "profit_take", "stop_loss", "vertical"
    side: int                 # +1 long, -1 short, 0 no trade
    label: int                # +1, -1, 0


@dataclass
class TripleBarrierResult:
    """Full triple barrier labeling result."""
    events: list[TripleBarrierEvent] = field(default_factory=list)
    labels: NDArray[np.int_] = field(default_factory=lambda: np.array([], dtype=int))
    sides: NDArray[np.int_] = field(default_factory=lambda: np.array([], dtype=int))
    returns: NDArray[np.float64] = field(default_factory=lambda: np.array([], dtype=float))
    barriers_hit: list[str] = field(default_factory=list)
    entry_indices: NDArray[np.int_] = field(default_factory=lambda: np.array([], dtype=int))
    exit_indices: NDArray[np.int_] = field(default_factory=lambda: np.array([], dtype=int))

    @property
    def n_events(self) -> int:
        return len(self.events)

    @property
    def label_distribution(self) -> dict[int, int]:
        """Count of each label value."""
        vals, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(vals.tolist(), counts.tolist()))


def triple_barrier_label(
    prices: np.ndarray,
    pt_level: float = 0.02,
    sl_level: float = 0.02,
    max_bars: int = 20,
    side: np.ndarray | None = None,
    vertical_first: bool = False,
    min_return: float = 0.0,
) -> TripleBarrierResult:
    """
    Triple barrier labeling (AFML ch. 3).

    For each bar, defines three barriers:
    - Upper (profit take): price rises pt_level %
    - Lower (stop loss): price drops sl_level %
    - Vertical: max_bars bars forward

    The first barrier touched determines the label.

    Args:
        prices: 1D price array.
        pt_level: Profit take as fraction (0.02 = 2%).
        sl_level: Stop loss as fraction (0.02 = 2%).
        max_bars: Maximum holding period in bars.
        side: Optional array of trade sides (+1/-1). If None, labels both sides.
        vertical_first: If True, vertical barrier has priority on ties.
        min_return: Minimum return to qualify as non-zero label.

    Returns:
        TripleBarrierResult with labels, sides, returns.
    """
    prices = np.asarray(prices, dtype=float).ravel()
    n = len(prices)

    events = []
    labels_list = []
    sides_list = []
    returns_list = []
    barriers_list = []
    entry_idx_list = []
    exit_idx_list = []

    for i in range(n - 1):
        p0 = prices[i]
        if p0 <= 0:
            continue

        # Define barriers
        upper = p0 * (1 + pt_level) if pt_level > 0 else float("inf")
        lower = p0 * (1 - sl_level) if sl_level > 0 else -float("inf")
        max_idx = min(i + max_bars, n - 1)

        barrier_hit = "vertical"
        exit_idx = max_idx

        # Scan forward for barrier touches
        for j in range(i + 1, max_idx + 1):
            pj = prices[j]

            if not vertical_first:
                if pj >= upper:
                    barrier_hit = "profit_take"
                    exit_idx = j
                    break
                elif pj <= lower:
                    barrier_hit = "stop_loss"
                    exit_idx = j
                    break
            else:
                # Vertical first: only check horizontal if we haven't reached max
                if j < max_idx:
                    if pj >= upper:
                        barrier_hit = "profit_take"
                        exit_idx = j
                        break
                    elif pj <= lower:
                        barrier_hit = "stop_loss"
                        exit_idx = j
                        break

        exit_price = prices[exit_idx]
        ret = (exit_price - p0) / p0

        # Determine side
        if side is not None and i < len(side):
            s = int(side[i])
        else:
            s = 1 if ret >= 0 else -1

        # Label based on barrier and side
        if barrier_hit == "profit_take":
            lbl = 1
        elif barrier_hit == "stop_loss":
            lbl = -1
        else:
            # Vertical: sign of return
            if abs(ret) < min_return:
                lbl = 0
            else:
                lbl = 1 if ret > 0 else -1

        event = TripleBarrierEvent(
            entry_idx=i,
            exit_idx=exit_idx,
            entry_price=p0,
            exit_price=exit_price,
            return_pct=ret,
            barrier_hit=barrier_hit,
            side=s,
            label=lbl,
        )
        events.append(event)
        labels_list.append(lbl)
        sides_list.append(s)
        returns_list.append(ret)
        barriers_list.append(barrier_hit)
        entry_idx_list.append(i)
        exit_idx_list.append(exit_idx)

    return TripleBarrierResult(
        events=events,
        labels=np.array(labels_list, dtype=int),
        sides=np.array(sides_list, dtype=int),
        returns=np.array(returns_list, dtype=float),
        barriers_hit=barriers_list,
        entry_indices=np.array(entry_idx_list, dtype=int),
        exit_indices=np.array(exit_idx_list, dtype=int),
    )


# ═══════════════════════════════════════════════════════════════════
#  Sample Uniqueness / Weights
# ═══════════════════════════════════════════════════════════════════

def compute_sample_uniqueness(
    entry_indices: np.ndarray,
    exit_indices: np.ndarray,
    n_bars: int,
) -> np.ndarray:
    """
    Compute average uniqueness of each sample (AFML ch. 4).

    A sample is unique if its label period doesn't overlap with others.
    Average uniqueness = mean(1/concurrent_count) over the label's bars.

    Args:
        entry_indices: Start index of each sample's label period.
        exit_indices: End index of each sample's label period.
        n_bars: Total number of bars.

    Returns:
        Array of uniqueness scores in [0, 1] for each sample.
    """
    n_samples = len(entry_indices)

    # Concurrency matrix: how many samples are active at each bar
    concurrency = np.zeros(n_bars, dtype=int)
    for i in range(n_samples):
        concurrency[entry_indices[i]:exit_indices[i] + 1] += 1

    # Average uniqueness per sample
    uniqueness = np.zeros(n_samples)
    for i in range(n_samples):
        start = entry_indices[i]
        end = exit_indices[i] + 1
        bars = concurrency[start:end]
        bars = bars[bars > 0]
        if len(bars) > 0:
            uniqueness[i] = np.mean(1.0 / bars)

    return uniqueness


def compute_sample_weights(
    entry_indices: np.ndarray,
    exit_indices: np.ndarray,
    n_bars: int,
    returns: np.ndarray | None = None,
    decay_halflife: int | None = None,
) -> np.ndarray:
    """
    Compute sample weights based on uniqueness and optional return attribution.

    Args:
        entry_indices: Start indices.
        exit_indices: End indices.
        n_bars: Total bars.
        returns: Optional return array for return-attribution weighting.
        decay_halflife: Optional time-decay half-life in bars.

    Returns:
        Normalized weight array.
    """
    uniqueness = compute_sample_uniqueness(entry_indices, exit_indices, n_bars)
    weights = uniqueness.copy()

    if returns is not None:
        # Weight by absolute return * uniqueness
        weights *= np.abs(returns)

    if decay_halflife is not None and decay_halflife > 0:
        # Apply exponential time decay
        n = len(weights)
        decay = np.exp(-np.log(2) * np.arange(n)[::-1] / decay_halflife)
        weights *= decay

    # Normalize
    total = weights.sum()
    if total > 0:
        weights /= total

    return weights


# ═══════════════════════════════════════════════════════════════════
#  Combinatorial Purged K-Fold CV
# ═══════════════════════════════════════════════════════════════════

class PurgedKFold:
    """
    Combinatorial purged cross-validation (AFML ch. 7).

    Standard k-fold leaks information when samples overlap in time.
    PurgedKFold:
    1. Removes (purges) from training any sample whose label period
       overlaps with the test set
    2. Applies an embargo: removes training samples within embargo_n
       bars after the test set (to avoid leaking through serial correlation)

    Usage:
        cv = PurgedKFold(n_splits=5, embargo_pct=0.01)
        for train_idx, test_idx in cv.split(X, entry_indices, exit_indices):
            model.fit(X[train_idx], y[train_idx])
            score = model.score(X[test_idx], y[test_idx])
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01,
    ):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(
        self,
        X: np.ndarray,
        entry_indices: np.ndarray,
        exit_indices: np.ndarray,
    ):
        """
        Generate purged train/test splits.

        Args:
            X: Feature matrix (n_samples, n_features).
            entry_indices: Sample start bar indices.
            exit_indices: Sample end bar indices.

        Yields:
            (train_indices, test_indices) tuples.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        fold_size = n_samples // self.n_splits

        # Total bars for embargo calculation
        max_bar = int(exit_indices.max()) if len(exit_indices) > 0 else n_samples
        embargo_n = int(max_bar * self.embargo_pct)

        for fold in range(self.n_splits):
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < self.n_splits - 1 else n_samples
            test_idx = indices[test_start:test_end]

            if len(test_idx) == 0:
                continue

            # Test set bar range
            test_bar_min = int(entry_indices[test_idx].min())
            test_bar_max = int(exit_indices[test_idx].max())

            # Build training set with purging
            train_idx = []
            for i in indices:
                if i in test_idx:
                    continue

                sample_start = int(entry_indices[i])
                sample_end = int(exit_indices[i])

                # Purge: skip if sample overlaps with test period
                if sample_end >= test_bar_min and sample_start <= test_bar_max:
                    continue

                # Embargo: skip if sample is too close after test period
                if sample_start > test_bar_max and sample_start <= test_bar_max + embargo_n:
                    continue

                train_idx.append(i)

            yield np.array(train_idx, dtype=int), test_idx

    def get_n_splits(self) -> int:
        return self.n_splits


# ═══════════════════════════════════════════════════════════════════
#  Feature Engineering
# ═══════════════════════════════════════════════════════════════════

def compute_momentum_features(
    prices: np.ndarray,
    windows: list[int] | None = None,
) -> dict[str, np.ndarray]:
    """
    Compute momentum features for ML pipeline.

    Features:
    - Returns at various windows
    - Volatility at various windows
    - RSI
    - Rate of change

    Args:
        prices: 1D price array.
        windows: Lookback windows (default: [5, 10, 21, 63]).

    Returns:
        Dict of feature name → 1D array.
    """
    if windows is None:
        windows = [5, 10, 21, 63]

    prices = np.asarray(prices, dtype=float).ravel()
    n = len(prices)
    features = {}

    # Log returns
    log_prices = np.log(np.maximum(prices, 1e-10))
    log_ret_1 = np.zeros(n)
    log_ret_1[1:] = log_prices[1:] - log_prices[:-1]
    features["log_return_1"] = log_ret_1

    for w in windows:
        # Returns over window
        ret = np.zeros(n)
        if w < n:
            ret[w:] = (prices[w:] - prices[:-w]) / np.maximum(prices[:-w], 1e-10)
        features[f"return_{w}"] = ret

        # Volatility over window
        vol = np.zeros(n)
        for i in range(w, n):
            vol[i] = np.std(log_ret_1[i - w + 1:i + 1])
        features[f"volatility_{w}"] = vol

    # RSI (14-period)
    rsi_period = 14
    if n > rsi_period:
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)

        avg_gain = np.zeros(n)
        avg_loss = np.zeros(n)

        if rsi_period < len(gain):
            avg_gain[rsi_period] = np.mean(gain[:rsi_period])
            avg_loss[rsi_period] = np.mean(loss[:rsi_period])

            for i in range(rsi_period + 1, n):
                avg_gain[i] = (avg_gain[i - 1] * (rsi_period - 1) + gain[i - 1]) / rsi_period
                avg_loss[i] = (avg_loss[i - 1] * (rsi_period - 1) + loss[i - 1]) / rsi_period

        rsi = np.zeros(n)
        for i in range(rsi_period, n):
            if avg_loss[i] > 0:
                rs = avg_gain[i] / avg_loss[i]
                rsi[i] = 100 - 100 / (1 + rs)
            else:
                rsi[i] = 100.0

        features["rsi_14"] = rsi

    return features


def build_feature_matrix(
    features: dict[str, np.ndarray],
    start_idx: int = 0,
) -> tuple[np.ndarray, list[str]]:
    """
    Stack feature dict into (n_samples, n_features) matrix.

    Args:
        features: Dict from compute_momentum_features().
        start_idx: Skip first N rows (warmup period).

    Returns:
        (feature_matrix, feature_names) tuple.
    """
    names = sorted(features.keys())
    arrays = [features[name][start_idx:] for name in names]
    matrix = np.column_stack(arrays)
    return matrix, names
