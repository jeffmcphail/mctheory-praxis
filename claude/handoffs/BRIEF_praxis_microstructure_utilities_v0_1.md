# Implementation Brief: Microstructure Utilities v0.1 (Khirman-inspired)

**Series:** praxis
**Priority:** P1 (foundational -- unblocks better signal detection across many downstream components)
**Mode:** B (code additions + tests + new DB reader, no live API calls beyond Binance trade reads)

**Estimated Scope:** S-M (60-90 min: ~120 lines of utility code + tests + one consumer + CLI subcommand)
**Estimated Cost:** none (purely local compute on already-collected data)
**Estimated Data Volume:** reads from existing `trades` table (growing live, currently ~10k+ rows post-v8.2.1 verification)
**Kill switch:** N/A (no long-running processes; one-shot CLI tests only)

---

## Context

Stas Khirman's "Institutional Options Trades Detector" (April 2026, DataDrivenInvestor) introduced a tiered-threshold whale detection pattern for US options flow. Three ideas from that work transfer directly to Praxis:

1. **MAD-based (Modified Z-Score) outlier detection** is strictly better than standard-deviation-based detection when the outlier you're trying to find contaminates the dispersion measure you're using to find it. This is the default failure mode of sigma-based detectors and it's silent.

2. **Adaptive sample-size switching** (MAD for n<30, median + sigma*std for n>=30) handles the reality that Praxis computes rolling windows over volatile activity regimes. A 5-minute window during BTC overnight lulls might have 50 trades; the same window during a macro event might have 5,000. Fixed-method detectors misbehave at both extremes.

3. **Tiered thresholds** (always-whale upper floor + never-whale lower floor + statistical test in the middle) encode real trading conviction better than a single statistical cutoff. The extreme zones don't need math; they need action.

These three ideas are cross-cutting. Once built, they get consumed by:
- v8.2.1 whale trade detector (first consumer, built in this Brief)
- v8.3 microstructure features (planned, 2-4 weeks out)
- Smart money tracker (Polymarket large-position detection)
- Convergence speed detector (Polymarket price-velocity outliers)
- Funding rate anomaly detector (unusual 8-hour funding moves)
- Any future order-book imbalance anomaly detector

Building the utilities as a standalone module -- with Khirman's first consumer (the whale detector) bundled in -- gives us the primitive and proves it against real data in one Brief.

This is v0.1 of a microstructure utilities library. Future Briefs (v0.2, v0.3) can extend with VPIN computation, order-flow-imbalance helpers, and other microstructure primitives as they're needed.

---

## Objective

Create `engines/microstructure_utils.py` containing three general-purpose statistics helpers, plus `engines/whale_detector.py` as the first consumer operating on the `trades` table. Add CLI subcommands for both. Ship with unit tests that verify the statistical properties (not just exit codes).

---

## Detailed Spec

### Step 1: Create `engines/microstructure_utils.py`

Three pure functions, no side effects, no DB access. This module is the stable primitive.

```python
"""
Microstructure utilities v0.1

General-purpose statistical helpers for detecting institutional/whale activity
in microstructure data (trades, order book snapshots, funding rates, etc.).

Primary references:
    Iglewicz, B. & Hoaglin, D.C. (1993). "How to Detect and Handle Outliers."
        ASQC Quality Press.
    Khirman, S. (2026). "How to Build an Institutional Options Trades Detector."
        DataDrivenInvestor.

Design principle: no hidden assumptions about sample size, distribution shape,
or data source. Every function is deterministic given its inputs.
"""

import numpy as np
import pandas as pd
from typing import Optional


def modified_zscore(series):
    """Compute the Modified Z-Score (Iglewicz & Hoaglin 1993) for each value.
    
    Uses MAD (Median Absolute Deviation) instead of standard deviation as the
    dispersion measure. The 0.6745 constant normalizes MAD to match sigma on
    normally-distributed data, so thresholds are on the same scale as standard
    Z-scores.
    
    Robustness: MAD is NOT contaminated by the outlier you're trying to detect,
    unlike standard deviation. This matters critically for small samples and
    for detecting rare large events in a stream of small ones.
    
    Args:
        series: 1D array-like of numeric values (pandas Series, numpy array,
            or list). NaN values are preserved in output at their positions.
    
    Returns:
        numpy array of same length; values are modified Z-scores.
        If MAD == 0 (all values at the median), returns zeros everywhere
        EXCEPT positions where the value != median, which get +inf/-inf.
        NaN positions in input produce NaN positions in output.
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    if n == 0:
        return np.array([])
    
    # Preserve NaN positions
    nan_mask = np.isnan(x)
    
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    
    if mad == 0:
        # Degenerate case: all finite values equal median
        # Return 0 for values at median, +/- inf for values not at median
        result = np.where(x == med, 0.0, np.sign(x - med) * np.inf)
        result[nan_mask] = np.nan
        return result
    
    result = 0.6745 * (x - med) / mad
    result[nan_mask] = np.nan
    return result


def adaptive_outlier_detector(
    series,
    sigma_z: float = 3.5,
    sigma: float = 3.0,
    n_threshold: int = 30,
) -> np.ndarray:
    """Detect outliers, auto-switching method based on sample size.
    
    Small samples (n < n_threshold): use Modified Z-Score (MAD-based).
        Rationale: with few data points, standard deviation is easily
        inflated by the outlier you're trying to find.
    
    Large samples (n >= n_threshold): use median + sigma * std.
        Rationale: with enough points, sigma stabilizes; using median as
        center (instead of mean) keeps the detector robust to skew.
    
    Args:
        series: 1D array-like of numeric values. NaN values are ignored
            for threshold computation but preserve their position in output.
        sigma_z: Modified Z-Score threshold for small-sample case. Default
            3.5 per Khirman; Iglewicz & Hoaglin recommend 3.5 as the
            conventional cutoff.
        sigma: Standard-deviation multiplier for large-sample case. Default
            3.0.
        n_threshold: Sample size at which the method switches. Default 30.
    
    Returns:
        numpy boolean array of same length. True = outlier (upper tail only).
        NaN positions in input return False.
    """
    x = np.asarray(series, dtype=float)
    n_finite = np.sum(~np.isnan(x))
    
    if n_finite == 0:
        return np.zeros(len(x), dtype=bool)
    
    if n_finite < n_threshold:
        mz = modified_zscore(x)
        # Upper tail only (Khirman: we care about anomalously LARGE invested)
        flags = np.where(np.isnan(mz), False, mz > sigma_z)
    else:
        med = np.nanmedian(x)
        std = np.nanstd(x)
        threshold = med + sigma * std
        flags = np.where(np.isnan(x), False, x > threshold)
    
    return flags


def tiered_threshold_detector(
    series,
    always_floor: float,
    never_floor: float,
    sigma_z: float = 3.5,
    sigma: float = 3.0,
    n_threshold: int = 30,
) -> np.ndarray:
    """Three-tier detection: always-flag, never-flag, middle uses adaptive test.
    
    Values >= always_floor are flagged regardless of statistics (trivially large).
    Values <= never_floor are NEVER flagged regardless of statistics (trivially small).
    Values in the middle zone use adaptive_outlier_detector.
    
    This pattern (from Khirman 2026) encodes real trading conviction:
    - Upper floor: "anything this big is definitely the signal, no math needed"
    - Lower floor: "anything this small is definitely noise, no matter what
      the statistical test says" (prevents many-small-trades summing to a
      large total from masquerading as a single whale event)
    
    Args:
        series: 1D array-like of numeric values (e.g. invested dollars per bar)
        always_floor: absolute upper threshold -- values >= this always flagged
        never_floor: absolute lower threshold -- values <= this never flagged
        sigma_z, sigma, n_threshold: passed to adaptive_outlier_detector
    
    Returns:
        numpy boolean array of same length.
    """
    x = np.asarray(series, dtype=float)
    
    always_flags = np.where(np.isnan(x), False, x >= always_floor)
    never_flags = np.where(np.isnan(x), False, x <= never_floor)
    
    # Apply adaptive test only to middle zone
    middle_mask = ~always_flags & ~never_flags & ~np.isnan(x)
    middle_values = np.where(middle_mask, x, np.nan)
    adaptive_flags = adaptive_outlier_detector(
        middle_values, sigma_z=sigma_z, sigma=sigma, n_threshold=n_threshold
    )
    
    # Combine: always-flag OR (middle-zone AND passes adaptive test)
    return always_flags | adaptive_flags
```

### Step 2: Create tests `tests/test_microstructure_utils.py`

Tests MUST verify statistical properties, not just "function returns without error." Specifically:

```python
import numpy as np
import pandas as pd
import pytest
from engines.microstructure_utils import (
    modified_zscore,
    adaptive_outlier_detector,
    tiered_threshold_detector,
)


class TestModifiedZScore:
    def test_normalization_matches_sigma_on_normal_data(self):
        """Modified Z-Score with 0.6745 factor should approximate standard Z
        on normally-distributed data (core claim of Iglewicz & Hoaglin)."""
        np.random.seed(42)
        data = np.random.randn(10000)
        mz = modified_zscore(data)
        # The std of modified z-scores on normal data should be close to 1.0
        assert 0.85 < np.std(mz) < 1.15, f"Got std={np.std(mz):.3f}"
    
    def test_contamination_resistance(self):
        """Classic scenario: one outlier contaminates sigma; MAD stays robust."""
        # 29 normal values + 1 massive outlier
        data = np.concatenate([np.random.randn(29), [100.0]])
        # Standard Z-score of the outlier
        std_z = (100.0 - np.mean(data)) / np.std(data)
        # Modified Z-score of the outlier
        mz = modified_zscore(data)
        mod_z_outlier = mz[-1]
        # Modified Z should detect the outlier much more strongly
        # (standard z gets deflated because the outlier inflates sigma)
        assert mod_z_outlier > std_z, (
            f"Modified Z {mod_z_outlier:.1f} should exceed std Z {std_z:.1f}"
        )
        # Modified Z should be in the dozens+ range; standard Z stuck around 5
        assert mod_z_outlier > 20
    
    def test_preserves_nan_positions(self):
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = modified_zscore(data)
        assert np.isnan(result[2])
        assert not np.isnan(result[0])
    
    def test_degenerate_all_equal(self):
        data = np.array([5.0, 5.0, 5.0, 5.0])
        result = modified_zscore(data)
        assert np.all(result == 0.0)
    
    def test_empty_input(self):
        result = modified_zscore(np.array([]))
        assert len(result) == 0


class TestAdaptiveOutlierDetector:
    def test_small_sample_uses_mad(self):
        """With n=20 and a single large outlier, MAD-based detection should fire
        where sigma-based would be inflated and miss."""
        np.random.seed(0)
        data = np.concatenate([np.random.randn(19), [50.0]])
        flags = adaptive_outlier_detector(data, sigma_z=3.5, n_threshold=30)
        # The outlier should be flagged
        assert flags[-1]
        # Most of the random values should NOT be flagged
        assert np.sum(flags[:-1]) <= 1
    
    def test_large_sample_uses_std(self):
        """With n=1000 normal values, the detector should flag very few (<1%)."""
        np.random.seed(1)
        data = np.random.randn(1000)
        flags = adaptive_outlier_detector(data, sigma=3.0, n_threshold=30)
        # At sigma=3 on normal data, expect ~0.13% flags; allow up to 1%
        assert np.sum(flags) / 1000 < 0.01
    
    def test_switch_at_n_threshold(self):
        """Behavior should differ at the boundary. This is a sanity test for
        the switching logic, not a precise assertion on detection counts."""
        np.random.seed(2)
        # Data that produces different results under MAD vs sigma
        data_small = np.concatenate([np.random.randn(29), [10.0]])
        data_large = np.concatenate([np.random.randn(59), [10.0]])
        flags_small = adaptive_outlier_detector(data_small, n_threshold=30)
        flags_large = adaptive_outlier_detector(data_large, n_threshold=30)
        # Both should flag the outlier but via different mechanisms
        assert flags_small[-1]
        assert flags_large[-1]


class TestTieredThresholdDetector:
    def test_always_floor_bypasses_statistics(self):
        """A value above always_floor should be flagged even if it wouldn't
        pass the statistical test."""
        # Data dominated by one huge value would normally have inflated sigma
        data = np.array([1_000_000.0] + [100.0] * 50)
        flags = tiered_threshold_detector(
            data, always_floor=500_000, never_floor=10_000
        )
        assert flags[0]  # 1M > always_floor, must flag
    
    def test_never_floor_overrides_statistics(self):
        """A statistically-outlying value below never_floor should NOT be flagged."""
        # Create data where small-relative-to-noise value would pass adaptive test
        data = np.array([0.0] * 100 + [50.0])  # 50 is small abs but huge relative
        flags = tiered_threshold_detector(
            data, always_floor=1000, never_floor=100,
            # use n_threshold high enough to force adaptive MAD path
            n_threshold=200
        )
        # The 50 would pass the adaptive test but never_floor=100 overrides
        assert not flags[-1]
    
    def test_middle_zone_uses_adaptive(self):
        """Values between floors should be tested by the adaptive detector."""
        np.random.seed(3)
        # Moderate values with one middle-zone outlier
        data = np.concatenate([
            np.random.randn(50) * 100 + 500,  # ~centered at 500
            [5000.0]  # outlier in middle zone
        ])
        flags = tiered_threshold_detector(
            data,
            always_floor=10_000,  # 5000 < always
            never_floor=50,       # 5000 > never
            sigma=3.0
        )
        assert flags[-1]  # should be flagged by the adaptive test
```

Run with: `pytest tests/test_microstructure_utils.py -v`

Expected result: 11 tests pass.

### Step 3: Create `engines/whale_detector.py`

First consumer of the new utilities. Reads from `trades` table, detects single-trade whales and aggregated-window whales.

```python
"""
Whale trade detector -- Praxis v8.2.2

First consumer of microstructure_utils. Detects institutional-size trades
in the `trades` table populated by PraxisTradesCollector.

Two detection modes:

1. Single-trade whales: individual trades where dollar value >= always_floor.
   These are "block trade" whales -- one entity deploying large capital in
   one market order.

2. Windowed-aggregate whales: rolling windows where total invested dollars
   pass the tiered threshold test. These catch sustained accumulation that
   doesn't manifest as a single block but IS institutional in aggregate.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np

from engines.microstructure_utils import (
    modified_zscore,
    adaptive_outlier_detector,
    tiered_threshold_detector,
)

# Asset-specific whale thresholds (USD)
# Tune these based on observed data distribution during initial calibration
DEFAULT_THRESHOLDS = {
    "BTC": {
        "single_trade_always": 500_000,   # $500k single market order = always whale
        "single_trade_never":    1_000,   # below $1k = ignore even if statistical
        "window_always":       2_000_000, # $2M in the window = always whale window
        "window_never":          10_000,  # below $10k in window = never whale
    },
    "ETH": {
        "single_trade_always": 200_000,
        "single_trade_never":      500,
        "window_always":         800_000,
        "window_never":            5_000,
    },
}


def detect_single_trade_whales(
    asset: str,
    conn: sqlite3.Connection,
    lookback_minutes: int = 60,
    thresholds: dict = None,
) -> pd.DataFrame:
    """Find individual trades exceeding the single_trade_always threshold.
    
    Args:
        asset: "BTC" or "ETH"
        conn: SQLite connection to crypto_data.db
        lookback_minutes: how far back to scan
        thresholds: override DEFAULT_THRESHOLDS for this asset
    
    Returns:
        DataFrame with columns: timestamp, datetime, price, amount,
        quote_amount, side, trade_id.
        Sorted by quote_amount descending (biggest whale first).
    """
    t = thresholds or DEFAULT_THRESHOLDS.get(asset, {})
    always = t.get("single_trade_always", 500_000)
    
    cutoff_ms = int(
        (datetime.now(tz=timezone.utc).timestamp() - lookback_minutes * 60) * 1000
    )
    
    query = """
        SELECT trade_id, timestamp, datetime, price, amount, quote_amount, side
        FROM trades
        WHERE asset = ?
          AND timestamp >= ?
          AND quote_amount >= ?
        ORDER BY quote_amount DESC
    """
    df = pd.read_sql(query, conn, params=(asset, cutoff_ms, always))
    return df


def detect_windowed_whales(
    asset: str,
    conn: sqlite3.Connection,
    lookback_minutes: int = 60,
    window_seconds: int = 30,
    thresholds: dict = None,
) -> pd.DataFrame:
    """Aggregate trades into time windows, detect whale windows via tiered threshold.
    
    Returns:
        DataFrame with columns: window_start, window_end, trade_count,
        total_invested, buy_invested, sell_invested, aggressor_imbalance,
        is_whale, detection_reason ("always_floor" or "statistical").
    """
    t = thresholds or DEFAULT_THRESHOLDS.get(asset, {})
    always = t.get("window_always", 2_000_000)
    never = t.get("window_never", 10_000)
    
    cutoff_ms = int(
        (datetime.now(tz=timezone.utc).timestamp() - lookback_minutes * 60) * 1000
    )
    
    query = """
        SELECT timestamp, quote_amount, side
        FROM trades
        WHERE asset = ? AND timestamp >= ?
        ORDER BY timestamp ASC
    """
    df = pd.read_sql(query, conn, params=(asset, cutoff_ms))
    if df.empty:
        return pd.DataFrame()
    
    # Bucket trades into windows
    df["window_start_ms"] = (df["timestamp"] // (window_seconds * 1000)) * (window_seconds * 1000)
    
    # Aggregate per window
    windows = df.groupby("window_start_ms").agg(
        trade_count=("quote_amount", "count"),
        total_invested=("quote_amount", "sum"),
        buy_invested=("quote_amount", lambda x: x[df.loc[x.index, "side"] == "buy"].sum()),
        sell_invested=("quote_amount", lambda x: x[df.loc[x.index, "side"] == "sell"].sum()),
    ).reset_index()
    
    windows["aggressor_imbalance"] = (
        (windows["buy_invested"] - windows["sell_invested"])
        / windows["total_invested"].where(windows["total_invested"] > 0, 1)
    )
    
    windows["window_start"] = pd.to_datetime(
        windows["window_start_ms"], unit="ms", utc=True
    )
    windows["window_end"] = windows["window_start"] + pd.Timedelta(seconds=window_seconds)
    
    # Apply tiered threshold
    flags = tiered_threshold_detector(
        windows["total_invested"].to_numpy(),
        always_floor=always,
        never_floor=never,
    )
    windows["is_whale"] = flags
    
    # Classify why each whale fired
    windows["detection_reason"] = np.where(
        windows["total_invested"] >= always, "always_floor",
        np.where(flags, "statistical", "not_whale")
    )
    
    whales = windows[windows["is_whale"]].copy()
    whales = whales.sort_values("total_invested", ascending=False)
    return whales[[
        "window_start", "window_end", "trade_count", "total_invested",
        "buy_invested", "sell_invested", "aggressor_imbalance",
        "detection_reason"
    ]]


def summarize_whales(single_df: pd.DataFrame, window_df: pd.DataFrame, asset: str):
    """Print a human-readable summary of whale detections."""
    print(f"\n=== Whale Detection Summary: {asset} ===")
    
    print(f"\nSingle-trade whales: {len(single_df)}")
    if len(single_df) > 0:
        total_single = single_df["quote_amount"].sum()
        buy_count = (single_df["side"] == "buy").sum()
        sell_count = (single_df["side"] == "sell").sum()
        print(f"  Total dollar volume: ${total_single:,.0f}")
        print(f"  Buy-initiated: {buy_count}, Sell-initiated: {sell_count}")
        print(f"  Top 5:")
        for _, row in single_df.head(5).iterrows():
            print(f"    {row['datetime']}  {row['side']:4s}  "
                  f"${row['quote_amount']:>12,.0f}  @ ${row['price']:>10,.2f}")
    
    print(f"\nWindowed whale events: {len(window_df)}")
    if len(window_df) > 0:
        print(f"  Top 5:")
        for _, row in window_df.head(5).iterrows():
            imb_pct = row["aggressor_imbalance"] * 100
            print(f"    {row['window_start'].strftime('%H:%M:%S')}  "
                  f"${row['total_invested']:>12,.0f}  "
                  f"{row['trade_count']:>4} trades  "
                  f"imb={imb_pct:+6.1f}%  ({row['detection_reason']})")
```

### Step 4: CLI subcommand in `engines/crypto_data_collector.py`

Add a `detect-whales` subcommand:

```python
p_dw = subs.add_parser("detect-whales",
    help="Detect whale trades/windows from the trades table.")
p_dw.add_argument("--asset", required=True, choices=["BTC", "ETH"])
p_dw.add_argument("--lookback", type=int, default=60,
    help="Lookback in minutes (default 60)")
p_dw.add_argument("--window-seconds", type=int, default=30,
    help="Aggregation window for windowed detection (default 30)")


def cmd_detect_whales(args):
    from engines.whale_detector import (
        detect_single_trade_whales,
        detect_windowed_whales,
        summarize_whales,
    )
    conn = sqlite3.connect(DB_PATH)
    try:
        single = detect_single_trade_whales(
            args.asset, conn, lookback_minutes=args.lookback
        )
        windowed = detect_windowed_whales(
            args.asset, conn,
            lookback_minutes=args.lookback,
            window_seconds=args.window_seconds,
        )
        summarize_whales(single, windowed, args.asset)
    finally:
        conn.close()
```

Wire it into the dispatch map alongside the existing `collect-trades`, `collect-order-book`, etc.

### Step 5: Manual verification run

After implementation, run this and paste output in retro:

```powershell
# Give the task some time to accumulate trade data first (already running)
python -m engines.crypto_data_collector detect-whales --asset BTC --lookback 60
python -m engines.crypto_data_collector detect-whales --asset ETH --lookback 60
```

Expected output shape:
- Some number of single-trade whales (likely 0-20 in a quiet hour; could be more during active markets)
- Some windowed whale events (should be rare -- hence the "whale" label)
- `detection_reason` column populated showing `always_floor` vs `statistical`

**Calibration note:** default thresholds may need adjustment after first data review. If the BTC always_floor of $500k never fires in an hour, Jeff has probably seen no whales; if it fires 200 times in an hour, either threshold needs raising OR we genuinely had a whale-heavy hour. Don't tune in-Brief -- report the raw observed distribution in the retro and let Chat decide the calibration pass.

---

## Progress Reporting (per CLAUDE_CODE_RULES.md rules 9-15)

- **T+0:** confirm trades table has enough rows to be meaningful (should be in the tens of thousands by now given the live collector)
- **After each Step completes:** brief status
- **Tests must all pass before moving on from Step 2:** report pass/fail count
- **Post verification:** report the observed whale detection counts for BTC + ETH over 60-min lookback + a small sample of the output

If any step takes > 15 min, out-of-cadence report.

---

## Acceptance Criteria

- [ ] `engines/microstructure_utils.py` created with three functions, pure (no side effects, no DB)
- [ ] All 11 unit tests in `tests/test_microstructure_utils.py` pass
- [ ] Modified Z-Score test verifies std of output on N(0,1) is in [0.85, 1.15] (not just that function runs)
- [ ] Contamination resistance test verifies MAD detects outlier that standard Z-score misses
- [ ] `engines/whale_detector.py` created with DEFAULT_THRESHOLDS, three functions
- [ ] `detect-whales` CLI subcommand wired into `crypto_data_collector.py` dispatch
- [ ] Manual verification run on BTC + ETH produces output with reasonable structure (whether or not whales actually fire -- calibration is Chat's job in follow-up)
- [ ] AST parse + ASCII check pass on all new/modified files
- [ ] No modifications to any collector code paths or existing DB tables
- [ ] Existing tests (if any) still pass

---

## Known Pitfalls

- **`side` field semantics.** In the trades table, `side = "buy"` means the taker was the buyer (they hit the ask; market pressure upward). `side = "sell"` means taker was the seller (hit the bid; pressure downward). `is_buyer_maker` is the inverse (True when seller hit the bid, because the buyer was the passive market-maker). Double-check the aggregation lambdas in `detect_windowed_whales` match the intended semantics.
- **`quote_amount` is nominal dollars at trade time.** This is the quote currency (USDT) value, not a USD value. For BTC/USDT and ETH/USDT they're ~1:1 with USD so this is fine, but documenting the assumption matters if we ever extend to non-stablecoin pairs.
- **Timezone.** `timestamp` is Unix milliseconds UTC. `cutoff_ms` calculation uses `datetime.now(tz=timezone.utc).timestamp()` -- explicit UTC. Do NOT use `datetime.now()` naive (local time), it'll be off by Jeff's timezone offset and produce empty results.
- **Pandas groupby on pre-filtered DataFrame.** The lambda in `detect_windowed_whales` uses `df.loc[x.index, "side"]` which assumes the original df's index is intact through the groupby. It is in this case because we don't reset index, but this is fragile -- if someone adds a `.reset_index()` call upstream, the lambda breaks. Comment to flag.
- **Threshold calibration is out of scope.** Default thresholds are engineered guesses from Khirman's $1M options whale scale, downscaled for crypto market size. First run may fire too much or too little; report the observed distribution, don't tune in-Brief.
- **Test `test_middle_zone_uses_adaptive` is slightly sensitive to seed.** The seed is fixed, but if pandas/numpy behavior ever changes for `groupby` or `nanmedian`, this test might need adjustment. It's testing a behavior, not a specific numerical output.
- **MAD == 0 degenerate case.** If all values are identical (e.g., during a market halt), MAD is zero and Modified Z-Score is undefined. The utility returns +/-inf for non-median values; adaptive_outlier_detector handles this gracefully. Test covers it.

---

## What NOT to change

- `trades` table schema (v8.2.1 landed, don't touch)
- `order_book_snapshots` table schema
- Any existing collector functions
- Any existing scheduled tasks
- `engines/intrabar_predictor.py` (frozen)
- Commit history (Chat will handle consolidation separately if requested)

---

## References

- Iglewicz, B. & Hoaglin, D.C. (1993). *How to Detect and Handle Outliers.* ASQC Quality Press. The authoritative reference for Modified Z-Score.
- Khirman, S. (2026). "How to Build an Institutional Options Trades Detector." DataDrivenInvestor. The specific application that inspired this Brief.
- `engines/crypto_data_collector.py` existing patterns (`collect_recent_trades`, dispatch structure)
- Existing trades table schema: v8.2.1 retro, section 2
- Existing workflow: `claude/WORKFLOW_MODES_PRAXIS.md`
- Progress rules: `claude/CLAUDE_CODE_RULES.md` rules 9-15

---

## Future work (not this Brief)

- **v0.2:** Add `order_flow_imbalance_rolling()` for order-book-depth rolling OFI computation (consumed by v8.3 microstructure features).
- **v0.2:** Add `vpin()` for VPIN (Volume-Synchronized PIN) computation as a toxic-flow indicator.
- **v0.3:** Apply MAD-based detection retroactively to existing Praxis outlier detection sites (funding rate anomaly detector, convergence detector, smart money tracker).
- **v0.3:** Add `tiered_threshold_detector` to the convergence detector (always-converging if velocity > X, never-converging if < Y, adaptive test between).
- **v0.4:** Extract the "cheap-scan -> expensive-analysis" pattern as an explicit utility/decorator after we have two concrete use cases.

These items accumulate in the "TODO (Praxis)" memory queue as candidate future Briefs -- not commitments.
