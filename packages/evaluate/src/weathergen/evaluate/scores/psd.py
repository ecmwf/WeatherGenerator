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
  Spherical Harmonic Transform on unstructured grids (octahedral, reduced
  Gaussian, regular lat-lon).  Ported from ``spectral_transforms.py`` to pure
  numpy using Legendre helpers from ``spectral_helpers.py``.

- **Path B – Zonal FFT PSD** (``method="zonal"``):
  1-D zonal FFT along the longitude dimension on a regular lat-lon grid.
  Absorbs the functions previously in ``example_extras/power_spectra/psd_calc.py``.
"""

from __future__ import annotations

import logging
import numpy as np

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Numpy-based Spherical Harmonic Transform (ported from spectral_helpers.py)
# ---------------------------------------------------------------------------


def _legendre_gauss_weights(n: int, a: float = -1.0, b: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return Legendre-Gauss nodes and weights on ``[a, b]``."""
    xlg, wlg = np.polynomial.legendre.leggauss(n)
    xlg = (b - a) * 0.5 * xlg + (b + a) * 0.5
    wlg = wlg * (b - a) * 0.5
    return xlg, wlg


def _legpoly(mmax: int, lmax: int, x: np.ndarray, inverse: bool = False) -> np.ndarray:
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
        assert 0 < truncation <= self.nlat, (
            f"Truncation {truncation} must be in (0, {self.nlat}]"
        )
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

    # -- internal FFT helpers -----------------------------------------------

    def _rfft_regular(self, x: np.ndarray) -> np.ndarray:
        """Batched real FFT for a *regular* grid.

        Parameters
        ----------
        x : np.ndarray, shape ``(..., grid)``

        Returns
        -------
        np.ndarray, complex, shape ``(..., nlat, nlon//2+1)``
        """
        nlon = self.lons_per_lat[0]
        return np.fft.rfft(x.reshape(*x.shape[:-1], self.nlat, nlon), norm="forward")

    def _rfft_reduced(self, x: np.ndarray) -> np.ndarray:
        """Per-ring real FFT for a *reduced* (variable-resolution) grid.

        Parameters
        ----------
        x : np.ndarray, shape ``(..., grid)``

        Returns
        -------
        np.ndarray, complex, shape ``(..., nlat, max_nlon//2+1)``
        """
        max_nlon = max(self.lons_per_lat)
        out_shape = (*x.shape[:-1], self.nlat, max_nlon // 2 + 1)
        out = np.zeros(out_shape, dtype=np.complex128)

        for i, (slon, nlon) in enumerate(zip(self.slon, self.lons_per_lat)):
            out[..., i, : nlon // 2 + 1] = np.fft.rfft(
                x[..., slon : slon + nlon], norm="forward"
            )
        return out

    # -- transform ---------------------------------------------------

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Compute the SHT.

        Parameters
        ----------
        x : np.ndarray, real, shape ``(..., grid)``

        Returns
        -------
        np.ndarray, complex, shape ``(..., L, M)`` where
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
# High-level SHT PSD
# ---------------------------------------------------------------------------


def sht_psd(
    data: np.ndarray,
    nlat: int,
    truncation: int | None = None,
    grid_type: str = "octahedral",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute PSD via Spherical Harmonic Transform.

    1. Forward SHT: spatial → spectral coefficients ``(l, m)``.
    2. PSD: L2-norm over ``m`` for each total wavenumber ``l``.

    Parameters
    ----------
    data : np.ndarray
        Spatial field with shape ``(n_points,)`` or ``(n_samples, n_points)``.
    nlat : int
        Number of latitudes in the grid.
    truncation : int | None
        Spectral truncation.  Defaults to ``nlat // 2 - 1``.
    grid_type : str
        One of ``"octahedral"``, ``"regular"``, ``"reduced"``.

    Returns
    -------
    wavenumbers : np.ndarray, shape ``(L,)``
        Total wavenumber indices ``0, 1, …, L-1``.
    psd : np.ndarray, shape ``(L,)``
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

    L = psd.shape[0]
    wavenumbers = np.arange(L, dtype=np.float64)

    return wavenumbers, psd


# ---------------------------------------------------------------------------
# Zonal FFT PSD (absorbed from psd_calc.py)
# ---------------------------------------------------------------------------


class ZonalPSD:
    """Zonal power spectral density via 1-D FFT along the longitude dimension.

    This class absorbs the functionality previously in
    ``example_extras/power_spectra/psd_calc.py``.
    """

    @staticmethod
    def psd_1d(ht: np.ndarray) -> np.ndarray:
        """Return the PSD for positive non-zero frequencies of an even-length signal.

        Parameters
        ----------
        ht : np.ndarray
            1-D real-valued signal (one latitude ring).

        Returns
        -------
        np.ndarray
            PSD for positive frequencies, length ``n // 2``.
        """
        n = len(ht)
        hf = np.fft.rfft(ht, norm="forward")
        power = np.abs(hf[1 : round(n / 2 + 1)]) ** 2
        power *= 2.0  # compensate for positive frequencies only
        return power

    @staticmethod
    def positive_frequencies(npoints: int, spacing_deg: float = 1.0) -> np.ndarray:
        """Return the positive frequencies for a signal of *npoints* evenly spaced points.

        Parameters
        ----------
        npoints : int
            Number of equally-spaced longitude points.
        spacing_deg : float
            Grid spacing in degrees.  Default is ``360 / npoints``.

        Returns
        -------
        np.ndarray
            Positive frequencies, length ``npoints // 2``.
        """
        freq = np.fft.fftfreq(npoints, d=spacing_deg)
        return np.abs(freq[1 : round(npoints / 2 + 1)])

    @classmethod
    def compute(
        cls,
        field_2d: np.ndarray,
    ) -> np.ndarray:
        """Compute the zonal PSD averaged over all latitude rows.

        Parameters
        ----------
        field_2d : np.ndarray
            2-D array of shape ``(nlat, nlon)``.

        Returns
        -------
        np.ndarray
            PSD of shape ``(nlon // 2,)``.
        """
        nlat, nlon = field_2d.shape
        psd_accum = np.zeros(nlon // 2)
        for row in field_2d:
            psd_accum += cls.psd_1d(row)
        psd_accum /= nlat
        return psd_accum


def zonal_psd(
    data: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    lat_range: tuple[float, float] = (-60.0, 60.0),
    regrid_resolution: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute zonal PSD using 1-D FFT along the longitude dimension.

    The input data is expected to be on a regular lat-lon grid already (or
    will be interpolated/regridded by the caller).

    Parameters
    ----------
    data : np.ndarray
        Field values.  If 1-D, interpreted as flattened ``(nlat * nlon,)``.
        If 2-D, interpreted as ``(nlat, nlon)`` or ``(n_samples, nlat * nlon)``.
        If 3-D, interpreted as ``(n_samples, nlat, nlon)``.
    lats : np.ndarray
        1-D array of latitude values (descending, length ``nlat``).
    lons : np.ndarray
        1-D array of longitude values (ascending, length ``nlon``).
    lat_range : tuple[float, float]
        Latitude bounds to restrict the computation to.
    regrid_resolution : float
        Grid spacing in degrees (used only for frequency calculation).

    Returns
    -------
    frequencies : np.ndarray
        Positive frequencies in cycles per degree, shape ``(nfreq,)``.
    psd : np.ndarray
        Power spectral density averaged over samples and latitude rows,
        shape ``(nfreq,)``.
    """
    nlat = len(lats)
    nlon = len(lons)

    # Reshape to (n_samples, nlat, nlon)
    if data.ndim == 1:
        data = data.reshape(1, nlat, nlon)
    elif data.ndim == 2:
        if data.shape == (nlat, nlon):
            data = data[np.newaxis, :, :]
        else:
            # (n_samples, nlat * nlon)
            data = data.reshape(data.shape[0], nlat, nlon)
    # data is now (n_samples, nlat, nlon)

    # Apply latitude mask
    lat_mask = (lats >= lat_range[0]) & (lats <= lat_range[1])
    data = data[:, lat_mask, :]
    nlon_sub = data.shape[2]

    # Compute PSD per sample and average
    psds = []
    for s in range(data.shape[0]):
        psds.append(ZonalPSD.compute(data[s]))
    psd = np.mean(psds, axis=0)

    spacing = 360.0 / nlon_sub if nlon_sub > 0 else regrid_resolution
    frequencies = ZonalPSD.positive_frequencies(nlon_sub, spacing_deg=spacing)

    return frequencies, psd


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def compute_psd_for_field(
    data: np.ndarray,
    method: str = "sht",
    nlat: int | None = None,
    lats: np.ndarray | None = None,
    lons: np.ndarray | None = None,
    lat_range: tuple[float, float] = (-60.0, 60.0),
    regrid_resolution: float = 1.0,
    sht_truncation: int | None = None,
    grid_type: str = "octahedral",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute PSD using the selected method.

    Parameters
    ----------
    data : np.ndarray
        Spatial field.  Shape depends on the method (see ``sht_psd`` / ``zonal_psd``).
    method : str
        ``"sht"`` for SHT-based PSD, ``"zonal"`` for zonal FFT PSD.
    nlat : int | None
        Number of latitudes (required for SHT method).
    lats, lons : np.ndarray | None
        Latitude / longitude coordinate arrays (required for zonal method).
    lat_range : tuple[float, float]
        Latitude bounds for the zonal method.
    regrid_resolution : float
        Grid spacing in degrees for the zonal method.
    sht_truncation : int | None
        Spectral truncation for SHT.
    grid_type : str
        Grid type for SHT (``"octahedral"``, ``"regular"``, ``"reduced"``).

    Returns
    -------
    x_values : np.ndarray
        Wavenumbers (SHT) or positive frequencies (zonal).
    psd : np.ndarray
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
    elif method == "zonal":
        if lats is None or lons is None:
            raise ValueError("lats and lons are required for method='zonal'")
        return zonal_psd(
            data=data,
            lats=lats,
            lons=lons,
            lat_range=lat_range,
            regrid_resolution=regrid_resolution,
        )
    else:
        raise ValueError(f"Unknown PSD method: {method!r}. Use 'sht' or 'zonal'.")


def compute_psd_score(
    gt: np.ndarray,
    p: np.ndarray,
    lats: np.ndarray | None,
    lons: np.ndarray | None,
    nlat: int | None,
    n_points: int,
    psd_method: str = "sht",
    psd_regrid_resolution: float = 1.0,
    psd_sht_truncation: int | None = None,
    lat_range: tuple[float, float] = (-60.0, 60.0),
) -> tuple[float, dict]:
    """Compute PSD for a pair of 2-D fields and return a scalar score + curves.

    This is the main entry point called from the Scores class. It handles NaN
    masking, calls ``compute_psd_for_field`` for both inputs, and computes a
    log-spectral MSE summary score.

    Parameters
    ----------
    gt, p : np.ndarray
        Ground truth and prediction arrays of shape ``(n_samples, n_points)``.
    lats, lons : np.ndarray | None
        Latitude / longitude arrays of length ``n_points`` (or None).
    nlat : int | None
        Number of latitudes (for SHT fallback).
    n_points : int
        Original number of spatial points (before NaN masking).
    psd_method : str
        ``"sht"`` or ``"zonal"``.
    psd_regrid_resolution : float
        Grid spacing for zonal method.
    psd_sht_truncation : int | None
        Spectral truncation for SHT.
    lat_range : tuple[float, float]
        Latitude bounds for zonal method.

    Returns
    -------
    score : float
        Log-spectral MSE scalar.
    attrs : dict
        Dict with keys ``"frequencies"``, ``"psd_target"``, ``"psd_prediction"``
        (lists for JSON serialization).
    """
    # Drop NaN columns (masked grid points)
    valid_mask = ~np.isnan(gt).all(axis=0)
    gt = gt[:, valid_mask]
    p = p[:, valid_mask]

    # Filter lat/lon to match valid points
    lats_valid = lats[valid_mask] if lats is not None and len(lats) == n_points else lats
    lons_valid = lons[valid_mask] if lons is not None and len(lons) == n_points else lons
    nlat_valid = len(np.unique(lats_valid)) if lats_valid is not None else nlat

    try:
        freq_gt, psd_gt = compute_psd_for_field(
            data=gt, method=psd_method, nlat=nlat_valid, lats=lats_valid, lons=lons_valid,
            lat_range=lat_range, regrid_resolution=psd_regrid_resolution,
            sht_truncation=psd_sht_truncation,
        )
        freq_p, psd_p = compute_psd_for_field(
            data=p, method=psd_method, nlat=nlat_valid, lats=lats_valid, lons=lons_valid,
            lat_range=lat_range, regrid_resolution=psd_regrid_resolution,
            sht_truncation=psd_sht_truncation,
        )
    except Exception:
        import logging
        logging.getLogger(__name__).exception("PSD computation failed, returning NaN.")
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
