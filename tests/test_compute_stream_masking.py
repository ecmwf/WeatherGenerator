# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for scripts/compute_stream_masking.py core functions."""

import sys
from pathlib import Path

import numpy as np

# Make the script importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from compute_stream_masking import (
    VarResult,
    compute_spatial_autocorr,
    correlation_length_to_hl_mask,
    format_groupings,
    format_results_table,
    generate_yaml_snippets,
    group_by_hl_mask,
    haversine_km,
)


class TestHaversine:
    def test_same_point_zero_distance(self):
        d = haversine_km(
            np.array([0.5]), np.array([1.0]),
            np.array([0.5]), np.array([1.0]),
        )
        np.testing.assert_allclose(d, 0.0, atol=1e-10)

    def test_known_distance(self):
        # Equator, 1 degree apart ~ 111.2 km
        d = haversine_km(
            np.array([0.0]), np.array([0.0]),
            np.array([0.0]), np.array([np.deg2rad(1.0)]),
        )
        assert 110 < d[0] < 112

    def test_antipodal(self):
        # North pole to south pole ~ 20015 km (half circumference)
        d = haversine_km(
            np.array([np.pi / 2]), np.array([0.0]),
            np.array([-np.pi / 2]), np.array([0.0]),
        )
        np.testing.assert_allclose(d, 6371 * np.pi, rtol=0.01)

    def test_vectorized(self):
        n = 1000
        lats = np.random.uniform(-np.pi / 2, np.pi / 2, n)
        lons = np.random.uniform(-np.pi, np.pi, n)
        d = haversine_km(lats, lons, lats, lons)
        np.testing.assert_allclose(d, 0.0, atol=1e-8)


class TestSpatialAutocorrelation:
    def _make_correlated_data(self, l_corr_km, n_points=5000, n_times=50, seed=42):
        """Generate synthetic data with known spatial correlation structure.

        Uses a simple exponential covariance model on a random set of points.
        """
        rng = np.random.default_rng(seed)

        # Random points on the sphere
        lats = np.arcsin(rng.uniform(-1, 1, n_points))
        lons = rng.uniform(-np.pi, np.pi, n_points)

        # For efficiency, generate data with spatial correlation by
        # smoothing random noise with a distance-dependent kernel
        # We use a simplified approach: sample at a coarser grid and interpolate
        data = np.zeros((n_times, n_points))

        for t in range(n_times):
            # Generate independent noise
            noise = rng.standard_normal(n_points)

            # Apply spatial smoothing: for each point, average with nearby points
            # This creates correlation that decays with distance
            # Use a simple approach: create correlation via linear combination
            # of different spatial scales
            n_scales = 3
            smoothed = np.zeros(n_points)
            for _ in range(n_scales):
                # Pick random reference points and create a smooth field
                n_refs = max(5, n_points // 100)
                ref_idx = rng.choice(n_points, n_refs, replace=False)
                for ri in ref_idx:
                    distances = haversine_km(
                        lats[ri] * np.ones(n_points), lons[ri] * np.ones(n_points),
                        lats, lons,
                    )
                    weights = np.exp(-distances / l_corr_km)
                    smoothed += weights * rng.standard_normal()

            data[t] = smoothed + 0.1 * noise  # Add small noise

        return data, lats, lons

    def test_long_correlation_detected(self):
        """Data with long correlation length should give large L_corr."""
        data, lats, lons = self._make_correlated_data(l_corr_km=2000, n_points=3000, n_times=30)
        l_corr, _, _ = compute_spatial_autocorr(
            data, lats, lons, n_sample_pairs=50_000, seed=42,
        )
        # Should detect correlation length > 500km (it's actually ~2000km)
        assert l_corr > 500, f"Expected L_corr > 500km for long-range field, got {l_corr:.0f}km"

    def test_short_vs_long_correlation_ordering(self):
        """Data with shorter correlation should yield smaller L_corr than longer."""
        data_long, lats, lons = self._make_correlated_data(
            l_corr_km=2000, n_points=3000, n_times=30, seed=42,
        )
        l_corr_long, _, _ = compute_spatial_autocorr(
            data_long, lats, lons, n_sample_pairs=50_000, seed=42,
        )

        # For "short" correlation: use mostly white noise with minimal smoothing
        rng = np.random.default_rng(99)
        data_short = rng.standard_normal((30, 3000))
        # Add a tiny bit of spatial structure so it's not pure noise
        data_short += 0.1 * data_long
        l_corr_short, _, _ = compute_spatial_autocorr(
            data_short, lats, lons, n_sample_pairs=50_000, seed=42,
        )

        assert l_corr_short < l_corr_long, (
            f"Expected short-range L_corr ({l_corr_short:.0f}km) < "
            f"long-range L_corr ({l_corr_long:.0f}km)"
        )

    def test_uncorrelated_data(self):
        """Pure noise should give short correlation length."""
        rng = np.random.default_rng(42)
        n_points, n_times = 3000, 30
        lats = np.arcsin(rng.uniform(-1, 1, n_points))
        lons = rng.uniform(-np.pi, np.pi, n_points)
        data = rng.standard_normal((n_times, n_points))

        l_corr, _, _ = compute_spatial_autocorr(
            data, lats, lons, n_sample_pairs=50_000, seed=42,
        )
        # White noise should give very short correlation
        assert l_corr < 800, f"Expected small L_corr for white noise, got {l_corr:.0f}km"

    def test_returns_correct_shapes(self):
        rng = np.random.default_rng(42)
        n_points, n_times = 1000, 10
        lats = np.arcsin(rng.uniform(-1, 1, n_points))
        lons = rng.uniform(-np.pi, np.pi, n_points)
        data = rng.standard_normal((n_times, n_points))

        l_corr, bin_centers, bin_corr = compute_spatial_autocorr(
            data, lats, lons, n_bins=30, n_sample_pairs=10_000, seed=42,
        )
        assert isinstance(l_corr, float)
        assert len(bin_centers) == 30
        assert len(bin_corr) == 30


class TestCorrelationLengthToHlMask:
    def test_large_corr_gives_low_hl(self):
        # Very large-scale field -> coarse mask blocks (low hl_mask)
        hl = correlation_length_to_hl_mask(2000, healpix_level=5, multiplier=1.5)
        assert hl <= 1, f"Expected hl_mask <= 1 for L_corr=2000km, got {hl}"

    def test_small_corr_gives_high_hl(self):
        # Small-scale field -> fine mask blocks (high hl_mask)
        hl = correlation_length_to_hl_mask(100, healpix_level=5, multiplier=1.5)
        assert hl >= 3, f"Expected hl_mask >= 3 for L_corr=100km, got {hl}"

    def test_minimum_hl_is_zero(self):
        hl = correlation_length_to_hl_mask(10000, healpix_level=5, multiplier=1.5)
        assert hl == 0

    def test_monotonic(self):
        """Larger correlation length should give equal or lower hl_mask."""
        prev_hl = 99
        for l_corr in [50, 100, 200, 500, 1000, 2000, 5000]:
            hl = correlation_length_to_hl_mask(l_corr, healpix_level=5, multiplier=1.5)
            assert hl <= prev_hl, "hl_mask should decrease as L_corr increases"
            prev_hl = hl


class TestGroupByHlMask:
    def test_basic_grouping(self):
        results = {
            "z_500": VarResult("z_500", 1200, 1, np.array([]), np.array([])),
            "z_850": VarResult("z_850", 1100, 1, np.array([]), np.array([])),
            "t_500": VarResult("t_500", 800, 2, np.array([]), np.array([])),
            "tp": VarResult("tp", 150, 4, np.array([]), np.array([])),
        }
        groups = group_by_hl_mask(results)

        assert 1 in groups
        assert 2 in groups
        assert 4 in groups
        assert set(groups[1]) == {"z_500", "z_850"}
        assert groups[2] == ["t_500"]
        assert groups[4] == ["tp"]

    def test_all_same_group(self):
        results = {
            "a": VarResult("a", 1000, 2, np.array([]), np.array([])),
            "b": VarResult("b", 900, 2, np.array([]), np.array([])),
        }
        groups = group_by_hl_mask(results)
        assert len(groups) == 1
        assert 2 in groups

    def test_empty_input(self):
        groups = group_by_hl_mask({})
        assert groups == {}


class TestFormatting:
    def test_format_results_table(self):
        results = {
            "z_500": VarResult("z_500", 1200, 1, np.array([]), np.array([])),
            "tp": VarResult("tp", 150, 4, np.array([]), np.array([])),
        }
        table = format_results_table(results)
        assert "z_500" in table
        assert "tp" in table
        assert "1200" in table

    def test_format_groupings(self):
        groups = {1: ["z_500", "z_850"], 4: ["tp"]}
        text = format_groupings(groups)
        assert "hl_mask=1" in text
        assert "hl_mask=4" in text
        assert "z_500" in text

    def test_generate_yaml_snippets(self):
        groups = {1: ["z_500"], 3: ["tp"]}
        yaml = generate_yaml_snippets(groups)
        assert "masking_override:" in yaml
        assert "hl_mask: 1" in yaml
        assert "hl_mask: 3" in yaml
