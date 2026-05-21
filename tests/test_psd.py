# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

"""Tests for weathergen.evaluate.scores.psd."""

from __future__ import annotations

import numpy as np
import pytest

from weathergen.evaluate.scores.psd import (
    SphericalHarmonicTransform,
    ZonalPSD,
    compute_psd_for_field,
    sht_psd,
    zonal_psd,
)


# ---------------------------------------------------------------------------
# ZonalPSD (absorbs psd_calc.py)
# ---------------------------------------------------------------------------


class TestZonalPSD:
    """Unit tests for the ZonalPSD class."""

    def test_psd_1d_even(self):
        """PSD of a pure sine should peak at the correct frequency bin."""
        n = 360
        freq_idx = 5  # wave number 5
        x = np.sin(2 * np.pi * freq_idx * np.arange(n) / n)
        power = ZonalPSD.psd_1d(x)
        assert power.shape == (n // 2,)
        # Peak should be at index freq_idx - 1 (psd starts at freq index 1)
        assert np.argmax(power) == freq_idx - 1

    def test_positive_frequencies(self):
        """Positive frequencies length and positivity."""
        npoints = 360
        freq = ZonalPSD.positive_frequencies(npoints, spacing_deg=1.0)
        assert len(freq) == npoints // 2
        assert np.all(freq > 0)

    def test_compute_2d(self):
        """ZonalPSD.compute averages PSD over latitude rows."""
        nlat, nlon = 10, 360
        field = np.random.default_rng(42).standard_normal((nlat, nlon))
        psd = ZonalPSD.compute(field)
        assert psd.shape == (nlon // 2,)
        assert np.all(psd >= 0)


# ---------------------------------------------------------------------------
# zonal_psd wrapper
# ---------------------------------------------------------------------------


class TestZonalPsdWrapper:
    """Tests for the zonal_psd() dispatch-level wrapper."""

    def test_basic_shape(self):
        """Output shape matches expected number of positive frequencies."""
        nlat, nlon = 90, 360
        lats = np.linspace(90, -90, nlat)
        lons = np.linspace(0, 359, nlon)
        data = np.random.default_rng(0).standard_normal((nlat, nlon))

        freq, psd = zonal_psd(data, lats, lons, lat_range=(-60, 60))
        assert freq.ndim == 1
        assert psd.ndim == 1
        assert len(freq) == len(psd)
        assert np.all(psd >= 0)

    def test_multi_sample(self):
        """Multi-sample input is averaged correctly."""
        nlat, nlon = 45, 180
        lats = np.linspace(90, -90, nlat)
        lons = np.linspace(0, 358, nlon)
        rng = np.random.default_rng(1)
        data = rng.standard_normal((3, nlat, nlon))  # 3 samples

        freq, psd = zonal_psd(data, lats, lons)
        assert freq.shape == psd.shape


# ---------------------------------------------------------------------------
# SphericalHarmonicTransform
# ---------------------------------------------------------------------------


class TestSphericalHarmonicTransform:
    """Tests for the pure-numpy SHT transform."""

    def test_regular_grid_shape(self):
        """SHT on a regular grid produces the correct output shape."""
        nlat = 32
        nlon = 64
        trunc = 15
        sht = SphericalHarmonicTransform(lons_per_lat=[nlon] * nlat, truncation=trunc)

        x = np.random.default_rng(7).standard_normal(sht.n_grid_points)
        coeffs = sht.transform(x)
        assert coeffs.shape == (trunc + 1, trunc + 1)
        assert np.iscomplexobj(coeffs)

    def test_octahedral_grid_shape(self):
        """SHT on an octahedral grid produces the correct output shape."""
        nlat = 64
        trunc = 31
        lons = [20 + 4 * i for i in range(nlat // 2)]
        lons = lons + list(reversed(lons))
        sht = SphericalHarmonicTransform(lons_per_lat=lons, truncation=trunc)

        x = np.random.default_rng(8).standard_normal(sht.n_grid_points)
        coeffs = sht.transform(x)
        assert coeffs.shape == (trunc + 1, trunc + 1)

    def test_constant_field(self):
        """A constant field should have energy only at wavenumber 0."""
        nlat = 32
        nlon = 64
        trunc = 15
        sht = SphericalHarmonicTransform(lons_per_lat=[nlon] * nlat, truncation=trunc)

        x = np.ones(sht.n_grid_points) * 42.0
        coeffs = sht.transform(x)
        # l=0, m=0 should dominate
        assert np.abs(coeffs[0, 0]) > 0
        # All other coefficients should be negligible
        mask = np.ones_like(coeffs, dtype=bool)
        mask[0, 0] = False
        assert np.allclose(coeffs[mask], 0, atol=1e-10)


# ---------------------------------------------------------------------------
# sht_psd
# ---------------------------------------------------------------------------


class TestShtPsd:
    """Tests for the sht_psd high-level function."""

    def test_output_shape(self):
        """sht_psd returns wavenumbers and PSD with matching shapes."""
        nlat = 64
        lons = [20 + 4 * i for i in range(nlat // 2)]
        lons = lons + list(reversed(lons))
        n_points = sum(lons)
        data = np.random.default_rng(9).standard_normal(n_points)

        wn, psd = sht_psd(data, nlat=nlat, grid_type="octahedral")
        assert wn.shape == psd.shape
        assert len(wn) == nlat // 2  # truncation default = nlat // 2 - 1 → L = nlat // 2
        assert np.all(psd >= 0)

    def test_multi_sample(self):
        """Multi-sample input is averaged."""
        nlat = 32
        nlon = 64
        n_points = nlat * nlon
        data = np.random.default_rng(10).standard_normal((4, n_points))

        wn, psd = sht_psd(data, nlat=nlat, grid_type="regular")
        assert wn.shape == psd.shape


# ---------------------------------------------------------------------------
# compute_psd_for_field dispatch
# ---------------------------------------------------------------------------


class TestComputePsdForField:
    """Tests for the dispatch function."""

    def test_sht_dispatch(self):
        """method='sht' calls sht_psd correctly."""
        nlat = 32
        nlon = 64
        data = np.random.default_rng(11).standard_normal(nlat * nlon)
        wn, psd = compute_psd_for_field(data, method="sht", nlat=nlat, grid_type="regular")
        assert len(wn) == len(psd)
        assert np.all(psd >= 0)

    def test_zonal_dispatch(self):
        """method='zonal' calls zonal_psd correctly."""
        nlat, nlon = 90, 360
        lats = np.linspace(90, -90, nlat)
        lons = np.linspace(0, 359, nlon)
        data = np.random.default_rng(12).standard_normal((nlat, nlon))

        freq, psd = compute_psd_for_field(
            data, method="zonal", lats=lats, lons=lons
        )
        assert len(freq) == len(psd)

    def test_invalid_method(self):
        """Unknown method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown PSD method"):
            compute_psd_for_field(np.zeros(10), method="invalid")

    def test_missing_nlat_for_sht(self):
        """method='sht' without nlat raises ValueError."""
        with pytest.raises(ValueError, match="nlat is required"):
            compute_psd_for_field(np.zeros(10), method="sht")

    def test_missing_lats_for_zonal(self):
        """method='zonal' without lats raises ValueError."""
        with pytest.raises(ValueError, match="lats and lons are required"):
            compute_psd_for_field(np.zeros(10), method="zonal")
