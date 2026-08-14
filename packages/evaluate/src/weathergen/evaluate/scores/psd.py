# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Power Spectral Density (PSD) computation.

Provides two PSD computation paths:

- **Path A – SHT-based PSD** (``method="sht"``):
  Spherical Harmonic Transform on separable grids (octahedral, reduced
  Gaussian, regular lat-lon).  Ported from anemoi.models ``spectral_transforms.py`` to pure
  numpy using Legendre helpers from anemoi.models ``spectral_helpers.py``.
  [anemoi.models.spectral_transforms]
  https://github.com/ecmwf/anemoi-core/blob/main/models/src/anemoi/models/layers/spectral_transforms.py
  [anemoi.models.spectral_helpers]
  https://github.com/ecmwf/anemoi-core/blob/main/models/src/anemoi/models/layers/spectral_helpers.py

- **Path B – FFT PSD** (``method="fft"``):
  1-D zonal FFT along the longitude dimension.  This method **requires a regular
  lat-lon grid** — i.e. the data must already live on a structured grid where
  every latitude ring has the same number of equally-spaced longitude points.
  If the input grid is not regular (e.g. octahedral reduced Gaussian), the
  function raises a ``ValueError``.  Re-gridding to a regular grid prior to FFT
  is deliberately not supported because the interpolation introduces spectral
  artefacts whose effect on the PSD is ill-defined.
  For non-regular grids, use the SHT method instead.
  Code base provided by the UKMet Office.

  The PSD is then computed row-by-row (per latitude ring) via 1-D real FFT
  and averaged over all latitude rows within the specified ``lat_range``.
"""

from __future__ import annotations

import logging

import numpy as np

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Numpy-based Spherical Harmonic Transform (ported from spectral_helpers.py)
# ---------------------------------------------------------------------------


def _legendre_gauss_weights(
    n: int, a: float = -1.0, b: float = 1.0
) -> tuple[np.typing.NDArray, np.typing.NDArray]:
    """Return Legendre-Gauss nodes and weights on ``[a, b]``."""
    xlg, wlg = np.polynomial.legendre.leggauss(n)
    xlg = (b - a) * 0.5 * xlg + (b + a) * 0.5
    wlg = wlg * (b - a) * 0.5
    return xlg, wlg


def _legpoly(
    mmax: int, lmax: int, x: np.typing.NDArray, inverse: bool = False
) -> np.typing.NDArray:
    """Compute associated Legendre polynomials.

    Returns shape ``(mmax+1, lmax+1, len(x))``.
    """
    nmax = max(mmax, lmax)
    vdm = np.zeros((nmax + 1, nmax + 1, len(x)), dtype=np.float64)

    norm_factor = np.sqrt(4 * np.pi)
    norm_factor = 1.0 / norm_factor if inverse else norm_factor
    vdm[0, 0, :] = norm_factor / np.sqrt(4 * np.pi)

    for n in range(1, nmax + 1):
        vdm[n - 1, n, :] = np.sqrt(2 * n + 1) * x * vdm[n - 1, n - 1, :]
        vdm[n, n, :] = np.sqrt((2 * n + 1) * (1 + x) * (1 - x) / 2 / n) * vdm[n - 1, n - 1, :]

    for n in range(2, nmax + 1):
        for m in range(0, n - 1):
            vdm[m, n, :] = (
                x * np.sqrt((2 * n - 1) / (n - m) * (2 * n + 1) / (n + m)) * vdm[m, n - 1, :]
                - np.sqrt((n + m - 1) / (n - m) * (2 * n + 1) / (2 * n - 3) * (n - m - 1) / (n + m))
                * vdm[m, n - 2, :]
            )

    return vdm[: mmax + 1, : lmax + 1]


class SphericalHarmonicTransform:
    """Spherical Harmonic Transform in pure numpy.

    Mirrors the ``SphericalHarmonicTransform`` from ``spectral_helpers.py`` in anemoi.models
    but operates on numpy arrays rather than torch tensors.

    Parameters
    ----------
    lons_per_lat : list[int]
        Number of longitude points on each latitude ring (pole to pole).
    truncation : int
        Maximum total wavenumber to retain.
    """

    def __init__(self, lons_per_lat: list[int], truncation: int) -> None:
        self.lons_per_lat = lons_per_lat
        self.nlat = len(lons_per_lat)
        self.truncation = truncation
        assert 0 < truncation <= self.nlat, f"Truncation {truncation} must be in (0, {self.nlat}]"
        self.n_grid_points = sum(lons_per_lat)

        # Offsets into the flattened grid for each latitude ring
        self.slon = [0] + list(np.cumsum(lons_per_lat))[:-1]

        # Whether all rings have the same number of points (regular grid)
        self._is_regular = len(set(lons_per_lat)) == 1

        # Precompute Gaussian latitudes + quadrature weights
        theta, weight = _legendre_gauss_weights(self.nlat)
        theta = np.flip(np.arccos(theta))

        # Associated Legendre polynomials  (m, l, lat)
        pct = _legpoly(truncation, truncation, np.cos(theta))

        # Pre-multiply by quadrature weights  → shape (m, l, lat)
        self.weight = np.einsum("mlk,k->mlk", pct, weight)

    # internal FFT helpers

    def _rfft_regular(self, x: np.typing.NDArray) -> np.typing.NDArray:
        """Batched real FFT for a *regular* grid.

        Parameters
        ----------
        x : np.typing.NDArray, shape ``(..., grid)``

        Returns
        -------
        np.typing.NDArray, complex, shape ``(..., nlat, nlon//2+1)``
        """
        nlon = self.lons_per_lat[0]
        return np.fft.rfft(x.reshape(*x.shape[:-1], self.nlat, nlon), norm="forward")

    def _rfft_reduced(self, x: np.typing.NDArray) -> np.typing.NDArray:
        """Per-ring real FFT for a *reduced* (variable-resolution) grid.

        Parameters
        ----------
        x : np.typing.NDArray, shape ``(..., grid)``

        Returns
        -------
        np.typing.NDArray, complex, shape ``(..., nlat, max_nlon//2+1)``
        """
        max_nlon = max(self.lons_per_lat)
        out_shape = (*x.shape[:-1], self.nlat, max_nlon // 2 + 1)
        out = np.zeros(out_shape, dtype=np.complex128)

        for i, (slon, nlon) in enumerate(zip(self.slon, self.lons_per_lat, strict=False)):
            out[..., i, : nlon // 2 + 1] = np.fft.rfft(x[..., slon : slon + nlon], norm="forward")
        return out

    # transform

    def transform(self, x: np.typing.NDArray) -> np.typing.NDArray:
        """Compute the SHT.

        Parameters
        ----------
        x : np.typing.NDArray, real, shape ``(..., grid)``

        Returns
        -------
        np.typing.NDArray, complex, shape ``(..., L, M)`` where
        ``L = M = truncation + 1``.
        """
        if self._is_regular:
            x_fft = self._rfft_regular(x)
        else:
            x_fft = self._rfft_reduced(x)

        x_fft = 2.0 * np.pi * x_fft

        real_part = x_fft[..., : self.truncation + 1].real
        imag_part = x_fft[..., : self.truncation + 1].imag

        rl = np.einsum("...km,mlk->...lm", real_part, self.weight)
        im = np.einsum("...km,mlk->...lm", imag_part, self.weight)

        return rl + 1j * im


class InverseSphericalHarmonicTransform:
    """Inverse Spherical Harmonic Transform in pure numpy.

    Reconstructs a spatial field from spectral coefficients (l, m).
    Mirrors the ``InverseSphericalHarmonicTransform`` from ``spectral_helpers.py``
    in anemoi.models but operates on numpy arrays.
    This is not needed for the PSD computation but it is included
    to verify that the forward and inverse transforms are consistent with each other.

    Parameters
    ----------
    lons_per_lat : list[int]
        Number of longitude points on each latitude ring (pole to pole).
    truncation : int
        Maximum total wavenumber.
    """

    def __init__(self, lons_per_lat: list[int], truncation: int) -> None:
        self.lons_per_lat = lons_per_lat
        self.nlat = len(lons_per_lat)
        self.truncation = truncation
        self.n_grid_points = sum(lons_per_lat)
        self._is_regular = len(set(lons_per_lat)) == 1

        # Gaussian latitudes (no quadrature weights needed for inverse)
        theta, _ = _legendre_gauss_weights(self.nlat)
        theta = np.flip(np.arccos(theta))

        # Associated Legendre polynomials with inverse=True
        self.pct = _legpoly(truncation, truncation, np.cos(theta), inverse=True)

    def _irfft_regular(self, x: np.typing.NDArray) -> np.typing.NDArray:
        """Inverse FFT for a regular grid.

        Parameters
        ----------
        x : np.typing.NDArray, complex, shape ``(..., nlat, M)``

        Returns
        -------
        np.typing.NDArray, real, shape ``(..., grid)``
        """
        nlon = self.lons_per_lat[0]
        spatial = np.fft.irfft(x, n=nlon, norm="forward")  # (..., nlat, nlon)
        return spatial.reshape(*spatial.shape[:-2], self.n_grid_points)

    def _irfft_reduced(self, x: np.typing.NDArray) -> np.typing.NDArray:
        """Per-ring inverse FFT for a reduced grid.

        Parameters
        ----------
        x : np.typing.NDArray, complex, shape ``(..., nlat, M)``

        Returns
        -------
        np.typing.NDArray, real, shape ``(..., grid)``
        """
        lead_shape = x.shape[:-2]
        out = np.zeros((*lead_shape, self.n_grid_points), dtype=np.float64)
        offset = 0
        for i, nlon in enumerate(self.lons_per_lat):
            ring = np.fft.irfft(x[..., i, :], n=nlon, norm="forward")
            out[..., offset : offset + nlon] = ring
            offset += nlon
        return out

    def transform(self, coeffs: np.typing.NDArray) -> np.typing.NDArray:
        """Compute the inverse SHT.

        Parameters
        ----------
        coeffs : np.typing.NDArray, complex, shape ``(..., L, M)``

        Returns
        -------
        np.typing.NDArray, real, shape ``(..., grid)``
        """
        # Inverse Legendre transform: (..., l, m) × (m, l, k) → (..., k, m)
        real_part = coeffs.real
        imag_part = coeffs.imag

        rl = np.einsum("...lm,mlk->...km", real_part, self.pct)
        im = np.einsum("...lm,mlk->...km", imag_part, self.pct)

        x_fourier = rl + 1j * im  # (..., nlat, M)

        # Inverse FFT per ring
        if self._is_regular:
            return self._irfft_regular(x_fourier)
        else:
            return self._irfft_reduced(x_fourier)


# ---------------------------------------------------------------------------
# Grid helpers for building lons_per_lat
# ---------------------------------------------------------------------------


def _octahedral_lons_per_lat(nlat: int) -> list[int]:
    """Return lons_per_lat for an octahedral reduced Gaussian grid."""
    half = [20 + 4 * i for i in range(nlat // 2)]
    return half + list(reversed(half))


def _regular_lons_per_lat(nlat: int) -> list[int]:
    """Return lons_per_lat for a regular lat-lon grid (nlon = 2*nlat)."""
    return [2 * nlat] * nlat


# ---------------------------------------------------------------------------
# Grid detection
# ---------------------------------------------------------------------------


def detect_grid_type(
    lats: np.typing.NDArray,
    lons: np.typing.NDArray,
    n_points: int,
) -> str | None:
    """Detect the grid type from latitude/longitude coordinates.

    Checks whether the point count matches known grid structures (octahedral
    reduced Gaussian or regular lat-lon). Returns ``None`` with a warning if
    the grid cannot be identified (e.g. regional subsets or non-standard grids).

    Parameters
    ----------
    lats : np.typing.NDArray
        Latitude values (per-point), length ``n_points``.
    lons : np.typing.NDArray
        Longitude values (per-point), length ``n_points``.
    n_points : int
        Total number of grid points.

    Returns
    -------
    str | None
        ``"octahedral"``, ``"regular"``, or ``None`` if detection fails.
    """
    unique_lats = np.unique(lats)
    nlat = len(unique_lats)

    # Check global extent
    lat_min, lat_max = unique_lats.min(), unique_lats.max()
    lat_span = lat_max - lat_min

    expected_span = 180.0 - 2 * (90.0 / nlat)  # approx span for a Gaussian grid
    if lat_span < 0.8 * expected_span:
        _logger.warning(
            f"Grid detection: latitude range [{lat_min:.1f}°, {lat_max:.1f}°] spans only "
            f"{lat_span:.1f}° (expected ~{expected_span:.1f}° for {nlat} latitudes). "
            f"PSD via SHT requires a global grid. Returning None."
        )
        return None

    # Check octahedral reduced Gaussian
    expected_oct = sum(_octahedral_lons_per_lat(nlat))
    if n_points == expected_oct:
        _logger.debug(f"Detected octahedral reduced Gaussian grid (nlat={nlat}).")
        return "octahedral"

    # Check regular lat-lon
    expected_reg = sum(_regular_lons_per_lat(nlat))
    if n_points == expected_reg:
        _logger.debug(f"Detected regular lat-lon grid (nlat={nlat}).")
        return "regular"

    # Check if all latitude rings have the same number of points (regular but non-standard ratio)
    unique_lons_global = np.unique(lons)
    if nlat * len(unique_lons_global) == n_points:
        _logger.debug(f"Detected regular grid (nlat={nlat}, nlon={len(unique_lons_global)}).")
        return "regular"

    _logger.warning(
        f"Grid detection: {n_points} points with {nlat} latitudes does not match "
        f"octahedral ({expected_oct}) or regular ({expected_reg}) grids. "
        f"The dataset may be regional or use an unsupported grid type."
        "PSD via SHT skipped."
    )
    return None


# ---------------------------------------------------------------------------
# High-level SHT PSD
# ---------------------------------------------------------------------------


def sht_psd(
    data: np.typing.NDArray,
    nlat: int,
    truncation: int | None = None,
    grid_type: str = "octahedral",
) -> tuple[np.typing.NDArray, np.typing.NDArray]:
    """Compute PSD via Spherical Harmonic Transform.

    1. Forward SHT: spatial → spectral coefficients ``(l, m)``.
    2. PSD: L2-norm over ``m`` for each total wavenumber ``l``.

    Parameters
    ----------
    data : np.typing.NDArray
        Spatial field with shape ``(n_points,)`` or ``(n_samples, n_points)``.
    nlat : int
        Number of latitudes in the grid.
    truncation : int | None
        Spectral truncation.  Defaults to ``nlat // 2 - 1``.
    grid_type : str
        One of ``"octahedral"``, ``"regular"``, ``"reduced"``.

    Returns
    -------
    wavenumbers : np.typing.NDArray, shape ``(L,)``
        Total wavenumber indices ``0, 1, …, L-1``.
    psd : np.typing.NDArray, shape ``(L,)``
        Power spectral density averaged over samples.
    """
    if data.ndim == 1:
        data = data[np.newaxis, :]
    n_samples, n_points = data.shape

    # Build the SHT for the appropriate grid
    if grid_type == "octahedral":
        lons_per_lat = _octahedral_lons_per_lat(nlat)
    elif grid_type == "regular":
        lons_per_lat = _regular_lons_per_lat(nlat)
    elif grid_type == "reduced":
        try:
            from anemoi.transform.grids.named import lookup
        except ImportError:
            raise ImportError(
                "anemoi.transform is required for grid_type='reduced'. "
                "Install: pip install anemoi-transform"
            ) from None
        lats = lookup("N320")["latitudes"]
        unique_lats = sorted(set(lats))
        lons_per_lat = [int((lats == lat).sum()) for lat in unique_lats]
    else:
        raise ValueError(f"Unknown grid_type: {grid_type!r}")

    trunc = truncation or nlat // 2 - 1
    sht = SphericalHarmonicTransform(lons_per_lat=lons_per_lat, truncation=trunc)

    assert n_points == sht.n_grid_points, (
        f"Input points={n_points} != expected grid points={sht.n_grid_points} "
        f"for grid_type={grid_type!r}, nlat={nlat}"
    )

    # SphericalHarmonicTransform.transform accepts (..., grid) → (..., L, M)
    # Pass (n_samples, n_points) directly.
    coeffs = sht.transform(data)  # (n_samples, L, M)

    # PSD = sum |coeffs|^2 over m for each total wavenumber l, averaged over samples
    psd_per_sample = np.sum(np.abs(coeffs) ** 2, axis=-1)  # (n_samples, L)
    psd = psd_per_sample.mean(axis=0)

    n_wavenumbers = psd.shape[0]
    wavenumbers = np.arange(n_wavenumbers, dtype=np.float64)

    return wavenumbers, psd


# ---------------------------------------------------------------------------
# FFT PSD (Credits to UK MetOffice)
# ---------------------------------------------------------------------------


def _fft_psd_calc(ht: np.typing.NDArray) -> np.typing.NDArray:
    """Return the PSD for positive non-zero frequencies of an even-length signal.

    Assumes *ht* has an even number of points.

    Parameters
    ----------
    ht : np.typing.NDArray
        1-D real-valued signal (one latitude ring).

    Returns
    -------
    np.typing.NDArray
        PSD for positive frequencies, length ``n // 2``.
    """
    n = len(ht)
    hf = np.fft.rfft(ht, norm="forward")
    power = np.abs(hf[1 : round(n / 2 + 1)]) ** 2
    power *= 2.0  # compensate for positive frequencies only
    return power


def _cubepsd(field_2d: np.typing.NDArray) -> np.typing.NDArray:
    """Compute PSD averaged over all latitude rows.

    Parameters
    ----------
    field_2d : np.typing.NDArray
        2-D array of shape ``(nlat, nlon)``.

    Returns
    -------
    np.typing.NDArray
        PSD of shape ``(nlon // 2,)``.
    """
    nlat, nlon = field_2d.shape
    field_psd = np.zeros(nlon // 2)
    for row in field_2d:
        field_psd += _fft_psd_calc(row)
    field_psd /= nlat
    return field_psd


def _calcposfreq(npoints: int, spacing_deg: float = 1.0) -> np.typing.NDArray:
    """Return the positive frequencies for a signal of *npoints* evenly spaced points.

    Parameters
    ----------
    npoints : int
        Number of equally-spaced longitude points.
    spacing_deg : float
        Grid spacing in degrees.

    Returns
    -------
    np.typing.NDArray
        Positive frequencies, length ``npoints // 2``.
    """
    freq = np.fft.fftfreq(npoints, d=spacing_deg)
    return np.abs(freq[1 : round(npoints / 2 + 1)])


def fft_psd(
    data: np.typing.NDArray,
    lats: np.typing.NDArray,
    lons: np.typing.NDArray,
    lat_range: tuple[float, float] = (-60.0, 60.0),
) -> tuple[np.typing.NDArray, np.typing.NDArray]:
    """Compute PSD using 1-D zonal FFT along the longitude dimension.

    This method requires a **regular lat-lon grid** where every latitude ring
    has the same number of equally-spaced longitude points.  If the input is
    not a regular grid, a ``ValueError`` is raised — use the SHT method instead.

    The PSD is computed row-by-row (per latitude ring) via 1-D real FFT and
    averaged over all latitude rows within the specified ``lat_range``.

    Parameters
    ----------
    data : np.typing.NDArray
        Field values.  Shape ``(n_samples, n_points)`` or ``(n_points,)``.
    lats : np.typing.NDArray
        Latitude values (per-point), length ``n_points``.
    lons : np.typing.NDArray
        Longitude values (per-point), length ``n_points``.
    lat_range : tuple[float, float]
        Latitude bounds to restrict the computation to.

    Returns
    -------
    frequencies : np.typing.NDArray
        Positive frequencies in cycles per degree, shape ``(nfreq,)``.
    psd : np.typing.NDArray
        Power spectral density averaged over samples and latitude rows,
        shape ``(nfreq,)``.

    Raises
    ------
    ValueError
        If the input grid is not a regular lat-lon grid.
    """

    # Ensure 2-D: (n_samples, n_points)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    n_samples, n_points = data.shape

    # Verify the grid is regular
    unique_lats = np.unique(lats)
    unique_lons = np.unique(lons)
    nlat, nlon = len(unique_lats), len(unique_lons)

    if nlat * nlon != n_points:
        raise ValueError(
            f"FFT PSD requires a regular lat-lon grid, but got {n_points} points "
            f"with {nlat} unique latitudes and {nlon} unique longitudes "
            f"(expected {nlat}×{nlon} = {nlat * nlon}). "
            f"Use psd_method='sht' for non-regular grids."
        )

    # Reshape to (n_samples, nlat, nlon) — points are assumed ordered lat-major
    data_3d = data.reshape(n_samples, nlat, nlon)

    # Apply latitude mask
    lat_axis = unique_lats
    lat_mask = (lat_axis >= lat_range[0]) & (lat_axis <= lat_range[1])
    data_3d = data_3d[:, lat_mask, :]
    nlon_sub = data_3d.shape[2]

    # Compute PSD per sample and average
    psds = []
    for s in range(data_3d.shape[0]):
        psds.append(_cubepsd(data_3d[s]))
    psd_result = np.mean(psds, axis=0)

    spacing = 360.0 / nlon_sub if nlon_sub > 0 else 1.0
    frequencies = _calcposfreq(nlon_sub, spacing_deg=spacing)
    return frequencies, psd_result


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def compute_psd_for_field(
    data: np.typing.NDArray,
    method: str = "sht",
    nlat: int | None = None,
    lats: np.typing.NDArray | None = None,
    lons: np.typing.NDArray | None = None,
    lat_range: tuple[float, float] = (-60.0, 60.0),
    sht_truncation: int | None = None,
    grid_type: str = "octahedral",
) -> tuple[np.typing.NDArray, np.typing.NDArray]:
    """Compute PSD using the selected method.

    Parameters
    ----------
    data : np.typing.NDArray
        Spatial field.  Shape depends on the method (see ``sht_psd`` / ``fft_psd``).
    method : str
        ``"sht"`` for SHT-based PSD, ``"fft"`` for FFT PSD.
    nlat : int | None
        Number of latitudes (required for SHT method).
    lats, lons : np.typing.NDArray | None
        Latitude / longitude coordinate arrays (required for fft method).
    lat_range : tuple[float, float]
        Latitude bounds for the fft method.
    sht_truncation : int | None
        Spectral truncation for SHT.
    grid_type : str
        Grid type for SHT (``"octahedral"``, ``"regular"``, ``"reduced"``).

    Returns
    -------
    x_values : np.typing.NDArray
        Wavenumbers (SHT) or positive frequencies (fft).
    psd : np.typing.NDArray
        Power spectral density.
    """
    if method == "sht":
        if nlat is None:
            raise ValueError("nlat is required for method='sht'")
        return sht_psd(
            data=data,
            nlat=nlat,
            truncation=sht_truncation,
            grid_type=grid_type,
        )
    elif method == "fft":
        if lats is None or lons is None:
            raise ValueError("lats and lons are required for method='fft'")
        return fft_psd(
            data=data,
            lats=lats,
            lons=lons,
            lat_range=lat_range,
        )
    else:
        raise ValueError(f"Unknown PSD method: {method!r}. Use 'sht' or 'fft'.")


def compute_psd_score(
    gt: np.typing.NDArray,
    p: np.typing.NDArray,
    lats: np.typing.NDArray | None,
    lons: np.typing.NDArray | None,
    nlat: int | None,
    n_points: int,
    psd_method: str = "sht",
    psd_sht_truncation: int | None = None,
    lat_range: tuple[float, float] = (-60.0, 60.0),
    grid_type: str | None = None,
) -> tuple[float, dict]:
    """Compute PSD for a pair of 2-D fields and return a scalar score + curves.

    This is the main entry point called from the Scores class. It handles NaN
    masking, calls ``compute_psd_for_field`` for both inputs, and computes a
    log-spectral MSE summary score.

    Parameters
    ----------
    gt, p : np.typing.NDArray
        Ground truth and prediction arrays of shape ``(n_samples, n_points)``.
    lats, lons : np.typing.NDArray | None
        Latitude / longitude arrays of length ``n_points`` (or None).
    nlat : int | None
        Number of latitudes (for SHT fallback).
    n_points : int
        Original number of spatial points (before NaN masking).
    psd_method : str
        ``"sht"`` or ``"fft"``.
    psd_sht_truncation : int | None
        Spectral truncation for SHT.
    lat_range : tuple[float, float]
        Latitude bounds for fft method.
    grid_type : str | None
        Pre-detected grid type (``"octahedral"``, ``"regular"``).
        When ``None``, the grid type is auto-detected from lats/lons.
        Pass a pre-computed value to avoid repeated detection across channels.

    Returns
    -------
    score : float
        Log-spectral MSE scalar.
    attrs : dict
        Dict with keys ``"frequencies"``, ``"psd_target"``, ``"psd_prediction"``
        (lists for JSON serialization).
    """
    # Handle NaN grid points (e.g. from regional masking).
    valid_mask = ~np.isnan(gt).all(axis=0)
    gt = gt[:, valid_mask]
    p = p[:, valid_mask]

    # Filter lat/lon to match valid points
    lats_valid = lats[valid_mask] if lats is not None and len(lats) == n_points else lats
    lons_valid = lons[valid_mask] if lons is not None and len(lons) == n_points else lons
    nlat_valid = len(np.unique(lats_valid)) if lats_valid is not None else nlat

    # Auto-detect grid type if not pre-computed by caller
    if psd_method == "sht":
        if lats_valid is None or lons_valid is None:
            _logger.warning("PSD (SHT): lats/lons required for grid detection. Skipping.")
            return np.nan, {}
        if grid_type is None:
            grid_type = detect_grid_type(lats_valid, lons_valid, gt.shape[-1])

        if grid_type == "octahedral":
            expected_pts = sum(_octahedral_lons_per_lat(nlat_valid))
        elif grid_type == "regular":
            expected_pts = sum(_regular_lons_per_lat(nlat_valid))
        else:
            expected_pts = None

        actual_pts = gt.shape[-1]
        if expected_pts is not None and actual_pts != expected_pts:
            _logger.warning(
                f"PSD (SHT): grid point mismatch ({actual_pts} vs expected {expected_pts} "
                f"for grid_type={grid_type!r}, nlat={nlat_valid}). SHT scores are only "
                f"available for the full (global/unmasked) grid. Skipping this region."
            )
            return np.nan, {}

    try:
        freq_gt, psd_gt = compute_psd_for_field(
            data=gt,
            method=psd_method,
            nlat=nlat_valid,
            lats=lats_valid,
            lons=lons_valid,
            lat_range=lat_range,
            sht_truncation=psd_sht_truncation,
            grid_type=grid_type,
        )
        freq_p, psd_p = compute_psd_for_field(
            data=p,
            method=psd_method,
            nlat=nlat_valid,
            lats=lats_valid,
            lons=lons_valid,
            lat_range=lat_range,
            sht_truncation=psd_sht_truncation,
            grid_type=grid_type,
        )
    except Exception:
        _logger.exception("PSD computation failed, returning NaN.")
        return np.nan, {}

    # Scalar summary: mean squared error of log10 PSD
    valid = (psd_gt > 0) & (psd_p > 0)
    if valid.any():
        log_mse = float(np.mean((np.log10(psd_p[valid]) - np.log10(psd_gt[valid])) ** 2))
    else:
        log_mse = np.nan

    attrs = {
        "frequencies": freq_gt.tolist(),
        "psd_target": psd_gt.tolist(),
        "psd_prediction": psd_p.tolist(),
    }

    return log_mse, attrs
