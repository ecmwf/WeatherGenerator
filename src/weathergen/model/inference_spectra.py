# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Angular power spectra for the inference-time ODE diagnostics.

Two spaces have to be put on a common footing:

- **physical**: the decoded fields live on the native o96 octahedral reduced Gaussian grid
  (40 320 points).  Handled by the evaluation package's ``sht_psd`` *unmodified*, so these
  curves use exactly the same estimator as the ``psd`` evaluation score.
- **latent**: the tokens live on a HEALPix grid (``cf.healpix_level`` -> ``nside``), which is
  iso-latitude but has neither Gauss-Legendre ring colatitudes nor Gauss-Legendre quadrature
  weights, so ``SphericalHarmonicTransform`` cannot be pointed at it.  :func:`healpix_sht_psd`
  therefore reuses the evaluation package's Legendre machinery (``_legpoly``) with HEALPix ring
  geometry and equal-area quadrature weights.

Sharing ``_legpoly`` is what makes the two comparable: the same analytic field sampled on o96
and on HEALPix nside 32 returns PSD values agreeing to ~5 significant figures, so latent and
physical spectra share both the ``l`` axis *and* the absolute scale.

Convention note (inherited from the evaluation package): the returned PSD is
``sum_m |a_lm|^2``, i.e. it is *not* divided by ``2l+1``.  White noise therefore rises like
``2l+1`` rather than being flat -- see :func:`white_noise_reference`.
"""

from __future__ import annotations

import functools
import logging

import numpy as np
from astropy_healpix.healpy import pix2ang, ring2nest

# Upstream estimator, used verbatim for the physical fields.  ``_legpoly`` is private but is the
# piece that fixes the normalisation convention; importing it is what keeps the latent spectra on
# the same scale as the physical ones.
from weathergen.evaluate.scores.psd import (
    _legpoly,
    _octahedral_lons_per_lat,
    detect_grid_type,
    sht_psd,
)

logger = logging.getLogger(__name__)


@functools.cache
def _healpix_ring_geometry(nside: int):
    """Ring decomposition of a HEALPix map, in RING ordering.

    Returns
    -------
    thetas : colatitude of each ring, ascending (north -> south)
    nlon : number of pixels on each ring
    start : index of each ring's first pixel in the RING-ordered map
    phi0 : azimuth of each ring's first pixel (HEALPix rings are *not* phase aligned)
    weight : quadrature weight in ``cos(theta)`` for each ring

    ``weight`` is the equal-area HEALPix quadrature: every pixel subtends ``4*pi/npix``, so a ring
    of ``nlon`` pixels covers ``d(cos theta) = 2*nlon/npix`` once the ``2*pi`` azimuthal integral
    is factored out (that ``2*pi`` is applied in :func:`healpix_sht_psd`, mirroring
    ``SphericalHarmonicTransform.transform``).
    """
    npix = 12 * nside**2
    theta, phi = pix2ang(nside=nside, ipix=np.arange(npix), nest=False)
    # RING ordering already groups pixels by ring with ascending colatitude.
    ring_id = np.r_[0, np.cumsum(np.abs(np.diff(theta)) > 1e-12)]
    nlon = np.bincount(ring_id)
    start = np.r_[0, np.cumsum(nlon)[:-1]]
    return theta[start], nlon, start, phi[start], 2.0 * nlon / npix


@functools.cache
def _nest_to_ring_index(nside: int) -> np.typing.NDArray:
    """Index array ``idx`` such that ``map_nested[idx]`` is the map in RING ordering."""
    return ring2nest(nside, np.arange(12 * nside**2))


def healpix_sht_psd(
    maps: np.typing.NDArray, nside: int, truncation: int | None = None, nested: bool = True
) -> tuple[np.typing.NDArray, np.typing.NDArray]:
    """Angular power spectrum of one or more HEALPix maps.

    Mirrors ``SphericalHarmonicTransform.transform`` + ``sht_psd`` from the evaluation package,
    with HEALPix ring geometry substituted for the Gauss-Legendre one.

    Parameters
    ----------
    maps : ``(npix,)`` or ``(n_maps, npix)``.  The model's latent tokens are indexed by the
        HEALPix **nested** index (``ang2pix(..., nest=True)`` in the tokenizer), hence the default.
    nside : HEALPix nside, i.e. ``2**cf.healpix_level``.
    truncation : maximum total wavenumber; defaults to ``2*nside``, beyond which the equal-area
        quadrature degrades.
    nested : whether ``maps`` is in nested ordering.

    Returns
    -------
    wavenumbers, psd : ``(truncation+1,)`` each; ``psd`` is averaged over ``n_maps``.
    """
    maps = np.atleast_2d(np.asarray(maps, dtype=np.float64))
    npix = 12 * nside**2
    if maps.shape[-1] != npix:
        msg = f"Expected {npix} pixels for nside={nside}, got {maps.shape[-1]}"
        raise ValueError(msg)

    truncation = int(truncation if truncation is not None else 2 * nside)
    if nested:
        maps = maps[:, _nest_to_ring_index(nside)]

    thetas, nlon, start, phi0, weight = _healpix_ring_geometry(nside)
    # (m, l, ring), pre-multiplied by the quadrature weight -- as in the upstream __init__.
    wgt = np.einsum("mlk,k->mlk", _legpoly(truncation, truncation, np.cos(thetas)), weight)

    # Per-ring real FFT.  Unlike the Gauss-Legendre grids upstream handles, HEALPix rings are not
    # phase aligned, so each ring's coefficients need the exp(-i*m*phi0) shift before they can be
    # accumulated across rings.
    coef = np.zeros((maps.shape[0], len(nlon), truncation + 1), dtype=np.complex128)
    for k, (s, n) in enumerate(zip(start, nlon, strict=True)):
        ring_fft = np.fft.rfft(maps[:, s : s + n], norm="forward")
        m = np.arange(min(truncation + 1, ring_fft.shape[-1]))
        coef[:, k, m] = ring_fft[:, m] * np.exp(-1j * m * phi0[k])
    coef *= 2.0 * np.pi

    # Complex einsum (upstream splits real/imag, which is equivalent only without the phase shift).
    alm = np.einsum("...km,mlk->...lm", coef, wgt)
    psd = np.sum(np.abs(alm) ** 2, axis=-1).mean(axis=0)
    return np.arange(truncation + 1, dtype=np.float64), psd


def canonical_grid_order(lats: np.typing.NDArray, lons: np.typing.NDArray) -> np.typing.NDArray:
    """Permutation putting scattered grid points into the ordering ``sht_psd`` expects.

    Upstream builds its rings from ``flip(arccos(leggauss_nodes))``, i.e. colatitude ascending =
    latitude descending (north to south), with longitude ascending from 0 within each ring.  The
    points reaching us come in dataset order, so sort rather than assume.
    """
    return np.lexsort((np.asarray(lons) % 360.0, -np.asarray(lats)))


def _detect_reduced_gaussian(lats: np.typing.NDArray, n_points: int) -> str | None:
    """Return ``"reduced"`` if the points are the ECMWF reduced Gaussian grid ``sht_psd`` supports.

    ``detect_grid_type`` (evaluate pkg) only knows octahedral and regular grids; ERA5's native grid
    is a classic reduced Gaussian (N320 = 542080 points / 640 latitudes) that matches neither.
    ``sht_psd``'s ``grid_type="reduced"`` path is N320-specific, so gate on nlat == 640 and confirm
    the exact point count via anemoi's grid table (already the dependency that path relies on).
    """
    nlat = len(np.unique(lats))
    if nlat != 640:  # sht_psd's reduced path is hard-wired to N320
        return None
    try:
        from anemoi.transform.grids.named import lookup

        expected = len(np.asarray(lookup("N320")["latitudes"]))
    except Exception as exc:  # noqa: BLE001 - anemoi missing / offline / unknown grid
        logger.warning(f"Reduced Gaussian (N320) PSD unavailable: {exc}")
        return None
    return "reduced" if n_points == expected else None


def physical_psd(
    values: np.typing.NDArray,
    lats: np.typing.NDArray,
    lons: np.typing.NDArray,
    truncation: int | None = None,
) -> tuple[np.typing.NDArray, np.typing.NDArray] | None:
    """Angular power spectrum of a field sampled on the native (o96) grid.

    Delegates to the evaluation package's ``sht_psd`` after restoring the canonical point order.
    Returns ``None`` (with a warning) when the point cloud is not a recognised global grid, e.g.
    a regional subset or a run with ``max_num_targets`` still subsampling the targets.
    """
    values = np.atleast_2d(np.asarray(values, dtype=np.float64))
    lats = np.asarray(lats)
    lons = np.asarray(lons)

    # detect_grid_type warns "PSD via SHT skipped" for any grid it doesn't know -- including ERA5's
    # reduced Gaussian, which we DO handle below -- so silence that one call to avoid a misleading
    # log; messaging is re-emitted here only if both detectors fail.
    psd_logger = logging.getLogger("weathergen.evaluate.scores.psd")
    _prev_level = psd_logger.level
    psd_logger.setLevel(logging.ERROR)
    try:
        grid_type = detect_grid_type(lats, lons, values.shape[-1])
    finally:
        psd_logger.setLevel(_prev_level)
    if grid_type is None:
        # ERA5's native reduced Gaussian N320 is recognised by neither octahedral nor regular
        # detection; sht_psd handles it via grid_type="reduced". Recognise it here instead of
        # silently skipping the physical spectrum.
        grid_type = _detect_reduced_gaussian(lats, values.shape[-1])
        if grid_type is None:
            logger.warning(
                f"Physical PSD skipped: {values.shape[-1]} points / {len(np.unique(lats))} "
                "latitudes is not a recognised global grid (octahedral, regular, or N320)."
            )
            return None

    nlat = len(np.unique(lats))
    expected = sum(_octahedral_lons_per_lat(nlat)) if grid_type == "octahedral" else None
    if expected is not None and values.shape[-1] != expected:
        logger.warning(
            f"Physical PSD skipped: {values.shape[-1]} points for nlat={nlat} does not match the "
            f"{expected} expected on an {grid_type} grid (target subsampling still active?)."
        )
        return None

    order = canonical_grid_order(lats, lons)
    return sht_psd(values[:, order], nlat=nlat, truncation=truncation, grid_type=grid_type)


def white_noise_reference(
    wavenumbers: np.typing.NDArray, variance: float = 1.0
) -> np.typing.NDArray:
    """PSD of spatially white noise in this convention: proportional to ``2l+1``.

    Because the estimator returns ``sum_m |a_lm|^2`` rather than a per-mode mean, white noise is a
    rising line, not a flat one.  Plotted as a reference so the pure-noise state at the start of
    the ODE is recognisable by shape.
    """
    return variance * (2.0 * np.asarray(wavenumbers) + 1.0)
