"""Spectral masking via spherical harmonics on HEALPix grids.

Masks selected frequency bands in spherical harmonic space, removing
specific spatial scales from input fields. Uses healpy's native HEALPix
SHT — no re-gridding or interpolation needed, only nested↔ring reordering.
"""

import logging

import healpy as hp
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Suppress healpy's verbose "Sigma is ..." logging from map2alm/alm2map
logging.getLogger("healpy").setLevel(logging.WARNING)


def forward_sht(
    map_nested: NDArray[np.floating], nside: int, lmax: int
) -> NDArray[np.complexfloating]:
    """Transform a HEALPix map (NESTED ordering) to spherical harmonic coefficients.

    Args:
        map_nested: Pixel values in NESTED ordering, shape (npix,).
        nside: HEALPix nside parameter.
        lmax: Maximum multipole degree.

    Returns:
        alm coefficients array.
    """
    map_ring = hp.reorder(map_nested, n2r=True)
    return hp.map2alm(map_ring, lmax=lmax, iter=10)


def inverse_sht(alm: NDArray[np.complexfloating], nside: int, lmax: int) -> NDArray[np.floating]:
    """Transform spherical harmonic coefficients back to a HEALPix map (NESTED ordering).

    Args:
        alm: Spherical harmonic coefficients.
        nside: HEALPix nside parameter.
        lmax: Maximum multipole degree.

    Returns:
        Pixel values in NESTED ordering, shape (npix,).
    """
    map_ring = hp.alm2map(alm, nside, lmax=lmax)
    return hp.reorder(map_ring, r2n=True)


def generate_spectral_mask_bands(
    lmax: int,
    max_num_bands: int = 4,
    max_log_fraction: float = 0.10,
    min_log_fraction: float = 0.01,
    rng: np.random.Generator | None = None,
) -> list[tuple[int, int]]:
    """Generate random spectral mask bands in log-space.

    Algorithm:
    1. Work in log-space: range = [log(1), log(lmax)], total = log(lmax).
    2. Budget = max_log_fraction * total.
    3. Cap num_bands so each band gets at least min_log_fraction of the log-range.
    4. Reserve min_width per band, distribute remainder via Dirichlet.
    5. For each band: sample random center uniformly in log-space, expand by half-width.
    6. Convert to integer l values, clamp to [1, lmax], merge overlaps.
    7. Verify total log-coverage <= budget.

    Args:
        lmax: Maximum multipole degree.
        max_num_bands: Maximum number of bands to generate.
        max_log_fraction: Maximum fraction of log-range to mask.
        min_log_fraction: Minimum fraction of log-range per band (default 1%).
        rng: Random number generator (for reproducibility).

    Returns:
        List of (l_start, l_end) tuples (inclusive bounds), sorted and non-overlapping.
    """
    if rng is None:
        rng = np.random.default_rng()

    if lmax < 2:
        return []

    total_log = np.log(lmax)
    budget = max_log_fraction * total_log
    min_width = min_log_fraction * total_log

    # Cap number of bands so each can meet the minimum width
    max_possible_bands = max(1, int(budget / min_width))
    num_bands = rng.integers(1, min(max_num_bands, max_possible_bands) + 1)

    # Reserve minimum width per band, distribute remainder via Dirichlet
    reserved = num_bands * min_width
    remaining = budget - reserved
    extra = rng.dirichlet(np.ones(num_bands)) * remaining
    band_widths = min_width + extra

    # Sample centers and compute bounds
    bands: list[tuple[int, int]] = []
    for width in band_widths:
        half_w = width / 2.0
        # Sample center uniformly in log-space so the band fits within [log(1), log(lmax)]
        lo_min = half_w  # log(1) = 0
        lo_max = total_log - half_w
        if lo_min > lo_max:
            center = total_log / 2.0
        else:
            center = rng.uniform(lo_min, lo_max)

        lo = center - half_w
        hi = center + half_w

        l_start = max(1, int(np.floor(np.exp(lo))))
        l_end = min(lmax, int(np.ceil(np.exp(hi))))

        if l_start <= l_end:
            bands.append((l_start, l_end))

    # Sort and merge overlapping bands
    bands.sort()
    merged: list[tuple[int, int]] = []
    for start, end in bands:
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    # Verify total log-coverage does not exceed budget
    total_coverage = sum(np.log(end + 1) - np.log(start) for start, end in merged)
    if total_coverage > budget * 1.5:
        # Trim last band(s) if we significantly exceed budget due to integer rounding
        while len(merged) > 1 and total_coverage > budget * 1.5:
            removed = merged.pop()
            total_coverage -= np.log(removed[1] + 1) - np.log(removed[0])

    return merged


def mask_alm_bands(
    alm: NDArray[np.complexfloating],
    lmax: int,
    bands: list[tuple[int, int]],
) -> NDArray[np.complexfloating]:
    """Zero out alm coefficients for l values in the specified bands.

    Args:
        alm: Spherical harmonic coefficients (modified in-place).
        lmax: Maximum multipole degree.
        bands: List of (l_start, l_end) inclusive bounds.

    Returns:
        Modified alm array (same object, modified in-place).
    """
    alm = alm.copy()
    for l_start, l_end in bands:
        for ell in range(l_start, l_end + 1):
            for m in range(0, ell + 1):
                idx = hp.Alm.getidx(lmax, ell, m)
                alm[idx] = 0.0
    return alm


def apply_spectral_masking(
    map_nested: NDArray[np.floating],
    nside: int,
    lmax: int,
    bands: list[tuple[int, int]],
) -> NDArray[np.floating]:
    """Apply spectral masking to a HEALPix map (NESTED ordering).

    Handles both single-channel (npix,) and multi-channel (C, npix) inputs.

    Args:
        map_nested: Pixel values in NESTED ordering, shape (npix,) or (C, npix).
        nside: HEALPix nside parameter.
        lmax: Maximum multipole degree.
        bands: List of (l_start, l_end) inclusive bounds to mask.

    Returns:
        Masked pixel values in NESTED ordering, same shape as input.
    """
    if not bands:
        return map_nested.copy()

    if map_nested.ndim == 1:
        alm = forward_sht(map_nested, nside, lmax)
        alm_masked = mask_alm_bands(alm, lmax, bands)
        return inverse_sht(alm_masked, nside, lmax)

    if map_nested.ndim == 2:
        result = np.empty_like(map_nested)
        for c in range(map_nested.shape[0]):
            alm = forward_sht(map_nested[c], nside, lmax)
            alm_masked = mask_alm_bands(alm, lmax, bands)
            result[c] = inverse_sht(alm_masked, nside, lmax)
        return result

    msg = f"Expected 1D or 2D input, got {map_nested.ndim}D"
    raise ValueError(msg)


def _best_nside(n_points: int) -> int:
    """Find the HEALPix nside whose npix is closest to n_points."""
    level = max(0, round(np.log(max(1, n_points) / 12) / np.log(4)))
    best_nside = 2**level
    best_diff = abs(12 * best_nside**2 - n_points)
    for lev in (level - 1, level + 1):
        if lev < 0:
            continue
        ns = 2**lev
        diff = abs(12 * ns**2 - n_points)
        if diff < best_diff:
            best_nside = ns
            best_diff = diff
    return best_nside


def apply_spectral_masking_binned(
    data: NDArray[np.floating],
    coords_deg: NDArray[np.floating],
    lmax: int,
    bands: list[tuple[int, int]],
) -> NDArray[np.floating]:
    """Apply spectral masking to data on an arbitrary grid via HEALPix binning.

    1. Bins data onto the HEALPix grid closest in size to the input.
    2. Applies SHT-based spectral masking on the HEALPix grid.
    3. Subtracts the removed frequency content from the original grid points.

    Accepts both numpy arrays and torch tensors; always returns numpy.

    Args:
        data: Shape (N, C) — N data points, C channels.
        coords_deg: Shape (N, 2) — (latitude, longitude) in degrees.
        lmax: Maximum multipole degree for SHT.
        bands: Frequency bands to zero out, list of (l_start, l_end) inclusive.

    Returns:
        Masked data as numpy array, same shape as input.
    """
    data = np.asarray(data)
    coords_deg = np.asarray(coords_deg)

    if not bands:
        return data.copy()

    n_points, n_channels = data.shape
    nside = _best_nside(n_points)
    npix = 12 * nside**2

    # Clamp lmax to what this nside supports
    lmax_eff = min(lmax, 3 * nside - 1)

    # Clamp bands to effective lmax
    clamped_bands = [
        (l_start, min(l_end, lmax_eff)) for l_start, l_end in bands if l_start <= lmax_eff
    ]
    if not clamped_bands:
        return data.copy()

    # Map native grid points to HEALPix pixels
    theta = np.deg2rad(90.0 - coords_deg[:, 0])  # colatitude
    phi = np.deg2rad(coords_deg[:, 1] % 360.0)
    pixel_indices = hp.ang2pix(nside, theta, phi, nest=True)

    result = data.copy()

    for c in range(n_channels):
        # Bin to HEALPix: average data per pixel
        healpix_map = np.zeros(npix, dtype=np.float64)
        counts = np.zeros(npix, dtype=np.int64)
        np.add.at(healpix_map, pixel_indices, data[:, c].astype(np.float64))
        np.add.at(counts, pixel_indices, 1)

        occupied = counts > 0
        healpix_map[occupied] /= counts[occupied]
        # Fill unoccupied pixels with global mean for SHT stability
        if not occupied.all():
            healpix_map[~occupied] = healpix_map[occupied].mean()

        # SHT → zero bands → iSHT
        alm = forward_sht(healpix_map, nside, lmax_eff)
        alm_masked = mask_alm_bands(alm, lmax_eff, clamped_bands)
        healpix_masked = inverse_sht(alm_masked, nside, lmax_eff)

        # Subtract removed content from each native grid point
        removed = healpix_map - healpix_masked
        result[:, c] -= removed[pixel_indices].astype(result.dtype)

    return result
