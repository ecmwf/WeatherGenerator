# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the inference-time ODE spectral diagnostics."""

import numpy as np
import pytest
from astropy_healpix.healpy import pix2ang
from scipy.special import sph_harm_y

from weathergen.evaluate.scores.psd import _legendre_gauss_weights, _octahedral_lons_per_lat
from weathergen.model.inference_spectra import (
    canonical_grid_order,
    healpix_sht_psd,
    physical_psd,
    white_noise_reference,
)

NSIDE = 32
NPIX = 12 * NSIDE**2
TRUNC = 2 * NSIDE


def _healpix_angles(nested: bool):
    return pix2ang(nside=NSIDE, ipix=np.arange(NPIX), nest=nested)


def _real_ylm(ell: int, m: int, theta, phi):
    y = sph_harm_y(ell, m, theta, phi)
    return y.real if m == 0 else np.sqrt(2) * y.real


@pytest.mark.parametrize(("ell", "m"), [(3, 2), (10, 0), (20, 7)])
def test_single_harmonic_is_isolated(ell: int, m: int) -> None:
    """A pure Y_lm must put essentially all its power at that l."""
    theta, phi = _healpix_angles(nested=True)
    _, psd = healpix_sht_psd(_real_ylm(ell, m, theta, phi), NSIDE, TRUNC)

    assert int(np.argmax(psd)) == ell
    off_peak = (psd.sum() - psd[ell]) / psd.sum()
    assert off_peak < 1e-3


def test_nested_and_ring_orderings_agree() -> None:
    """The nest->ring reindex must be applied; otherwise the spectrum is silently scrambled."""
    theta_n, phi_n = _healpix_angles(nested=True)
    theta_r, phi_r = _healpix_angles(nested=False)

    _, psd_nested = healpix_sht_psd(_real_ylm(7, 3, theta_n, phi_n), NSIDE, TRUNC, nested=True)
    _, psd_ring = healpix_sht_psd(_real_ylm(7, 3, theta_r, phi_r), NSIDE, TRUNC, nested=False)

    np.testing.assert_allclose(psd_nested, psd_ring, rtol=1e-10, atol=1e-14)


def test_ignoring_the_reindex_is_detectably_wrong() -> None:
    """Guard the guard: reading a nested map as if it were ring order must change the answer."""
    theta, phi = _healpix_angles(nested=True)
    field = _real_ylm(7, 3, theta, phi)

    _, correct = healpix_sht_psd(field, NSIDE, TRUNC, nested=True)
    _, scrambled = healpix_sht_psd(field, NSIDE, TRUNC, nested=False)

    assert int(np.argmax(correct)) == 7
    assert not np.allclose(correct, scrambled, rtol=1e-3)


def test_white_noise_follows_the_2l_plus_1_reference() -> None:
    """In this convention (sum over m, no 1/(2l+1)) white noise rises like 2l+1, not flat."""
    rng = np.random.default_rng(0)
    ell, psd = healpix_sht_psd(rng.standard_normal((64, NPIX)), NSIDE, TRUNC)

    ratio = psd[1:] / white_noise_reference(ell[1:])
    # Flat ratio => the measured spectrum has the 2l+1 shape.
    assert ratio.std() / ratio.mean() < 0.15


def _o96_grid():
    nlat = 192
    lons_per_lat = _octahedral_lons_per_lat(nlat)
    nodes, _ = _legendre_gauss_weights(nlat)
    theta = np.flip(np.arccos(nodes))
    theta_pts = np.concatenate([np.full(n, t) for t, n in zip(theta, lons_per_lat, strict=True)])
    phi_pts = np.concatenate([2 * np.pi * np.arange(n) / n for n in lons_per_lat])
    return nlat, theta_pts, phi_pts


def test_latent_and_physical_share_one_normalisation() -> None:
    """The point of reusing _legpoly: one analytic field, two grids, same PSD.

    Without this, the latent and physical panels would silently use different y-scales.
    """
    modes = [(4, 1, 1.0), (11, 5, 0.6), (25, 3, 0.3)]

    def field(theta, phi):
        return sum(amp * _real_ylm(ell, m, theta, phi) for ell, m, amp in modes)

    theta_hp, phi_hp = _healpix_angles(nested=True)
    _, psd_hp = healpix_sht_psd(field(theta_hp, phi_hp), NSIDE, TRUNC)

    nlat, theta_pts, phi_pts = _o96_grid()
    lats = 90.0 - np.degrees(theta_pts)
    lons = np.degrees(phi_pts)
    result = physical_psd(field(theta_pts, phi_pts), lats, lons, truncation=TRUNC)
    assert result is not None
    _, psd_o96 = result

    for ell, _, _ in modes:
        assert psd_hp[ell] == pytest.approx(psd_o96[ell], rel=1e-3)
    assert psd_hp.sum() == pytest.approx(psd_o96.sum(), rel=1e-3)


def test_physical_psd_is_order_independent() -> None:
    """Points arrive in dataset order, so the estimator must sort them itself."""
    nlat, theta_pts, phi_pts = _o96_grid()
    lats = 90.0 - np.degrees(theta_pts)
    lons = np.degrees(phi_pts)
    values = _real_ylm(6, 2, theta_pts, phi_pts)

    rng = np.random.default_rng(1)
    shuffle = rng.permutation(values.size)

    _, psd = physical_psd(values, lats, lons, truncation=TRUNC)
    _, psd_shuffled = physical_psd(values[shuffle], lats[shuffle], lons[shuffle], truncation=TRUNC)

    np.testing.assert_allclose(psd, psd_shuffled, rtol=1e-10, atol=1e-14)


def test_canonical_order_is_north_to_south_then_east() -> None:
    lats = np.array([-10.0, 45.0, 45.0, 80.0])
    lons = np.array([0.0, 200.0, 10.0, 5.0])
    np.testing.assert_array_equal(canonical_grid_order(lats, lons), [3, 2, 1, 0])


def test_subsampled_grid_is_refused_not_guessed() -> None:
    """With max_num_targets still active the point cloud is not a grid; must return None."""
    _, theta_pts, phi_pts = _o96_grid()
    rng = np.random.default_rng(2)
    keep = rng.choice(theta_pts.size, size=20000, replace=False)
    lats = 90.0 - np.degrees(theta_pts[keep])
    lons = np.degrees(phi_pts[keep])

    assert physical_psd(_real_ylm(6, 2, theta_pts[keep], phi_pts[keep]), lats, lons) is None
