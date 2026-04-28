"""
Microstructure utilities v0.1

General-purpose statistical helpers for detecting institutional/whale activity
in microstructure data (trades, order book snapshots, funding rates, etc.).

Primary references:
    Iglewicz, B. and Hoaglin, D.C. (1993). "How to Detect and Handle Outliers."
        ASQC Quality Press.
    Khirman, S. (2026). "How to Build an Institutional Options Trades Detector."
        DataDrivenInvestor.

Design principle: no hidden assumptions about sample size, distribution shape,
or data source. Every function is deterministic given its inputs. No side
effects, no DB access, no I/O.
"""

import numpy as np


def modified_zscore(series):
    """Compute the Modified Z-Score (Iglewicz and Hoaglin 1993) for each value.

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
        If MAD == 0 (all values at the median), returns zeros for values
        equal to the median, +/- inf for values not at median. NaN positions
        in input produce NaN positions in output.
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    if n == 0:
        return np.array([])

    nan_mask = np.isnan(x)

    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))

    if mad == 0:
        # Degenerate case: all finite values equal median
        result = np.where(x == med, 0.0, np.sign(x - med) * np.inf)
        result[nan_mask] = np.nan
        return result

    result = 0.6745 * (x - med) / mad
    result[nan_mask] = np.nan
    return result


def adaptive_outlier_detector(series, sigma_z=3.5, sigma=3.0, n_threshold=30):
    """Detect upper-tail outliers, auto-switching method based on sample size.

    Small samples (n < n_threshold): use Modified Z-Score (MAD-based).
        Rationale: with few data points, standard deviation is easily
        inflated by the outlier you're trying to find.

    Large samples (n >= n_threshold): use median + sigma * std.
        Rationale: with enough points, sigma stabilizes; using median as
        center (instead of mean) keeps the detector robust to skew.

    Args:
        series: 1D array-like of numeric values. NaN values are ignored for
            threshold computation but preserve their position in output.
        sigma_z: Modified Z-Score threshold for small-sample case. Default
            3.5 per Khirman; Iglewicz and Hoaglin recommend 3.5 as the
            conventional cutoff.
        sigma: Standard-deviation multiplier for large-sample case. Default 3.0.
        n_threshold: Sample size at which the method switches. Default 30.

    Returns:
        numpy boolean array of same length. True = upper-tail outlier.
        NaN positions in input return False.
    """
    x = np.asarray(series, dtype=float)
    n_finite = int(np.sum(~np.isnan(x)))

    if n_finite == 0:
        return np.zeros(len(x), dtype=bool)

    if n_finite < n_threshold:
        mz = modified_zscore(x)
        flags = np.where(np.isnan(mz), False, mz > sigma_z)
    else:
        med = np.nanmedian(x)
        std = np.nanstd(x)
        threshold = med + sigma * std
        flags = np.where(np.isnan(x), False, x > threshold)

    return np.asarray(flags, dtype=bool)


def tiered_threshold_detector(
    series,
    always_floor,
    never_floor,
    sigma_z=3.5,
    sigma=3.0,
    n_threshold=30,
):
    """Three-tier detection: always-flag, never-flag, middle uses adaptive test.

    Values >= always_floor are flagged regardless of statistics.
    Values <= never_floor are NEVER flagged regardless of statistics.
    Values in the middle zone use adaptive_outlier_detector.

    This pattern (from Khirman 2026) encodes real trading conviction:
    - Upper floor: "anything this big is the signal, no math needed"
    - Lower floor: "anything this small is noise, regardless of statistics"
      (prevents many-small-trades summing to a large total from masquerading
      as a single whale event)

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

    # Apply adaptive test only to middle zone (mask others with NaN)
    middle_mask = ~always_flags & ~never_flags & ~np.isnan(x)
    middle_values = np.where(middle_mask, x, np.nan)
    adaptive_flags = adaptive_outlier_detector(
        middle_values, sigma_z=sigma_z, sigma=sigma, n_threshold=n_threshold)

    return np.asarray(always_flags | adaptive_flags, dtype=bool)
