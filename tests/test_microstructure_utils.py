"""Tests for engines.microstructure_utils.

These tests verify statistical properties, not just that functions return
without error.
"""

import numpy as np
import pytest

from engines.microstructure_utils import (
    modified_zscore,
    adaptive_outlier_detector,
    tiered_threshold_detector,
)


class TestModifiedZScore:
    def test_normalization_matches_sigma_on_normal_data(self):
        """Modified Z-Score with 0.6745 factor should approximate standard Z
        on normally-distributed data (core claim of Iglewicz and Hoaglin)."""
        np.random.seed(42)
        data = np.random.randn(10000)
        mz = modified_zscore(data)
        # The std of modified z-scores on normal data should be close to 1.0
        assert 0.85 < np.std(mz) < 1.15, f"Got std={np.std(mz):.3f}"

    def test_contamination_resistance(self):
        """Classic scenario: one outlier contaminates sigma; MAD stays robust."""
        np.random.seed(0)
        data = np.concatenate([np.random.randn(29), [100.0]])
        std_z = (100.0 - np.mean(data)) / np.std(data)
        mz = modified_zscore(data)
        mod_z_outlier = mz[-1]
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
        assert flags[-1]
        assert np.sum(flags[:-1]) <= 1

    def test_large_sample_uses_std(self):
        """With n=1000 normal values, the detector should flag very few (<1%)."""
        np.random.seed(1)
        data = np.random.randn(1000)
        flags = adaptive_outlier_detector(data, sigma=3.0, n_threshold=30)
        # At sigma=3 on normal data, expect ~0.13% flags; allow up to 1%
        assert np.sum(flags) / 1000 < 0.01

    def test_switch_at_n_threshold(self):
        """Behavior should differ at the boundary; both paths flag an outlier."""
        np.random.seed(2)
        data_small = np.concatenate([np.random.randn(29), [10.0]])
        data_large = np.concatenate([np.random.randn(59), [10.0]])
        flags_small = adaptive_outlier_detector(data_small, n_threshold=30)
        flags_large = adaptive_outlier_detector(data_large, n_threshold=30)
        assert flags_small[-1]
        assert flags_large[-1]


class TestTieredThresholdDetector:
    def test_always_floor_bypasses_statistics(self):
        """A value above always_floor should be flagged even if it wouldn't
        pass the statistical test."""
        data = np.array([1_000_000.0] + [100.0] * 50)
        flags = tiered_threshold_detector(
            data, always_floor=500_000, never_floor=10_000,
        )
        assert flags[0]

    def test_never_floor_overrides_statistics(self):
        """A statistically-outlying value below never_floor should NOT be flagged."""
        # 100 zeros + one 50: the 50 is a massive outlier statistically
        # but 50 <= never_floor, so the tiered detector must suppress it.
        data = np.array([0.0] * 100 + [50.0])
        flags = tiered_threshold_detector(
            data, always_floor=1000, never_floor=100,
            # High n_threshold forces the adaptive-MAD path (not sigma path)
            n_threshold=200,
        )
        assert not flags[-1]

    def test_middle_zone_uses_adaptive(self):
        """Values between floors should be tested by the adaptive detector."""
        np.random.seed(3)
        data = np.concatenate([
            np.random.randn(50) * 100 + 500,  # ~centered at 500
            [5000.0],  # outlier in middle zone
        ])
        flags = tiered_threshold_detector(
            data,
            always_floor=10_000,
            never_floor=50,
            sigma=3.0,
        )
        assert flags[-1]
