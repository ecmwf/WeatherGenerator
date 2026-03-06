"""
Spherical Harmonic Transform helpers.

Copied (with minor formatting) from the anemoi-core PR #788:
https://github.com/ecmwf/anemoi-core/pull/788
Original file: models/src/anemoi/models/layers/spectral_helpers.py

(C) Copyright 2025 Anemoi contributors.
Licensed under the Apache Licence Version 2.0.
"""

import logging

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Legendre polynomial helpers
# ---------------------------------------------------------------------------


def legendre_gauss_weights(n: int, a: float = -1.0, b: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    r"""Return Legendre-Gauss nodes and weights on interval [a, b].

    Parameters
    ----------
    n : int
        Number of quadrature points.
    a : float
        Left endpoint (default -1).
    b : float
        Right endpoint (default 1).

    Returns
    -------
    xlg : np.ndarray
        Legendre-Gauss nodes.
    wlg : np.ndarray
        Legendre-Gauss weights.
    """
    xlg, wlg = np.polynomial.legendre.leggauss(n)
    xlg = (b - a) * 0.5 * xlg + (b + a) * 0.5
    wlg = wlg * (b - a) * 0.5
    return xlg, wlg


def legpoly(
    mmax: int,
    lmax: int,
    x: np.ndarray,
    inverse: bool = False,
) -> np.ndarray:
    r"""Compute (-1)^m c^l_m P^l_m(x) for associated Legendre polynomials.

    Result shape: ``(mmax, lmax, len(x))``.

    Parameters
    ----------
    mmax : int
        Maximum zonal wavenumber + 1.
    lmax : int
        Maximum total wavenumber + 1.
    x : np.ndarray
        Evaluation points in [-1, 1].
    inverse : bool
        If True, invert the normalisation (for the inverse Legendre transform).

    Notes
    -----
    Derived from torch-harmonics. Method follows Schaeffer (2013), Rapp (1982).
    """
    nmax = max(mmax, lmax)
    vdm = np.zeros((nmax, nmax, len(x)), dtype=np.float64)

    norm_factor = np.sqrt(4 * np.pi)
    norm_factor = 1.0 / norm_factor if inverse else norm_factor
    vdm[0, 0, :] = norm_factor / np.sqrt(4 * np.pi)

    # Fill diagonal and lower diagonal
    for n in range(1, nmax):
        vdm[n - 1, n, :] = np.sqrt(2 * n + 1) * x * vdm[n - 1, n - 1, :]
        vdm[n, n, :] = np.sqrt((2 * n + 1) * (1 + x) * (1 - x) / 2 / n) * vdm[n - 1, n - 1, :]

    # Fill remaining upper-triangle values
    for n in range(2, nmax):
        for m in range(0, n - 1):
            vdm[m, n, :] = (
                x * np.sqrt((2 * n - 1) / (n - m) * (2 * n + 1) / (n + m)) * vdm[m, n - 1, :]
                - np.sqrt((n + m - 1) / (n - m) * (2 * n + 1) / (2 * n - 3) * (n - m - 1) / (n + m))
                * vdm[m, n - 2, :]
            )

    vdm = vdm[:mmax, :lmax]
    return vdm


# ---------------------------------------------------------------------------
# Forward SHT
# ---------------------------------------------------------------------------


class SphericalHarmonicTransform(Module):
    r"""Forward spherical harmonic transform: grid → spectral coefficients.

    Works on both regular grids (same nlon every latitude) and reduced grids
    (variable nlon per latitude).

    Parameters
    ----------
    lons_per_lat : list[int]
        Number of longitudinal points on each latitude ring (pole to pole).
    lmax : int | None
        Maximum total wavenumber + 1 (defaults to nlat).
    mmax : int | None
        Maximum zonal wavenumber + 1 (defaults to nlat).

    Notes
    -----
    Inspired by the SHT in Nvidia's torch-harmonics.
    """

    def __init__(
        self,
        lons_per_lat: list[int],
        lmax: int | None = None,
        mmax: int | None = None,
    ) -> None:
        super().__init__()

        nlat = len(lons_per_lat)
        self.lmax = lmax or nlat
        self.mmax = mmax or nlat
        self.nlat = nlat
        self.lons_per_lat = lons_per_lat
        self.n_grid_points = sum(self.lons_per_lat)

        # Offsets into the flattened grid dimension
        self.slon = [0] + list(np.cumsum(self.lons_per_lat))[:-1]

        # Padding so every rFFT output has the same length
        self.rlon = [max(self.lons_per_lat) // 2 - nlon // 2 for nlon in self.lons_per_lat]

        # Pick efficient rfft strategy
        if len(set(self.lons_per_lat)) > 1:
            LOGGER.info("SphericalHarmonicTransform: Using rfft_rings_reduced")
            self.rfft_rings = self.rfft_rings_reduced
        else:
            LOGGER.info("SphericalHarmonicTransform: Using rfft_rings_regular")
            self.rfft_rings = self.rfft_rings_regular

        # Gaussian quadrature nodes and weights
        theta, weight = legendre_gauss_weights(nlat)
        theta = np.flip(np.arccos(theta))

        # Precompute associated Legendre polynomials × quadrature weights
        pct = legpoly(self.mmax, self.lmax, np.cos(theta))
        pct = torch.from_numpy(pct)
        weight = torch.from_numpy(weight)
        weight = torch.einsum("mlk, k -> mlk", pct, weight)

        self.register_buffer("weight", weight, persistent=False)

    def rfft_rings_reduced(self, x: Tensor) -> Tensor:
        """Real-to-complex FFT on each latitude ring of a reduced grid.

        Parameters
        ----------
        x : Tensor   shape [..., grid]

        Returns
        -------
        Tensor   shape [..., nlat, max_nlon//2+1]   (complex)
        """
        output_tensor = torch.zeros(
            *x.shape[:-1],
            self.nlat,
            max(self.lons_per_lat) // 2 + 1,
            device=x.device,
            dtype=torch.complex64 if x.dtype == torch.float32 else torch.complex128,
        )
        for i, (slon, nlon) in enumerate(zip(self.slon, self.lons_per_lat)):
            output_tensor[..., i, : nlon // 2 + 1] = torch.fft.rfft(
                x[..., slon : slon + nlon], norm="forward"
            )
        return output_tensor

    def rfft_rings_regular(self, x: Tensor) -> Tensor:
        """Real-to-complex FFT on each latitude ring of a regular grid.

        Parameters
        ----------
        x : Tensor   shape [..., grid]

        Returns
        -------
        Tensor   shape [..., nlat, nlon//2+1]   (complex)
        """
        return torch.fft.rfft(
            x.reshape(*x.shape[:-1], self.nlat, self.lons_per_lat[0]),
            norm="forward",
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward SHT: grid → spectral coefficients.

        Parameters
        ----------
        x : Tensor   shape [..., grid]

        Returns
        -------
        Tensor   shape [..., lmax, mmax]   (complex)
        """
        x = 2.0 * torch.pi * self.rfft_rings(x)
        x = torch.view_as_real(x)

        rl = torch.einsum("...km, mlk -> ...lm", x[..., : self.mmax, 0], self.weight.to(x.dtype))
        im = torch.einsum("...km, mlk -> ...lm", x[..., : self.mmax, 1], self.weight.to(x.dtype))

        x = torch.stack((rl, im), -1)
        x = torch.view_as_complex(x)
        return x


# ---------------------------------------------------------------------------
# Inverse SHT
# ---------------------------------------------------------------------------


class InverseSphericalHarmonicTransform(Module):
    r"""Inverse spherical harmonic transform: spectral coefficients → grid.

    Parameters
    ----------
    lons_per_lat : list[int]
        Number of longitudinal points on each latitude ring (pole to pole).
    lmax : int | None
        Maximum total wavenumber + 1 (defaults to nlat).
    mmax : int | None
        Maximum zonal wavenumber + 1 (defaults to nlat).

    Notes
    -----
    Inspired by the SHT in Nvidia's torch-harmonics.
    """

    def __init__(
        self,
        lons_per_lat: list[int],
        lmax: int | None = None,
        mmax: int | None = None,
    ) -> None:
        super().__init__()

        nlat = len(lons_per_lat)
        self.lmax = lmax or nlat
        self.mmax = mmax or nlat
        self.nlat = nlat
        self.lons_per_lat = lons_per_lat
        self.n_grid_points = sum(self.lons_per_lat)

        # Pick efficient irfft strategy
        if len(set(self.lons_per_lat)) > 1:
            LOGGER.info("InverseSphericalHarmonicTransform: Using irfft_rings_reduced")
            self.irfft_rings = self.irfft_rings_reduced
        else:
            LOGGER.info("InverseSphericalHarmonicTransform: Using irfft_rings_regular")
            self.irfft_rings = self.irfft_rings_regular

        # Gaussian quadrature latitudes (no weights needed for inverse)
        theta, _ = legendre_gauss_weights(nlat)
        theta = np.flip(np.arccos(theta))

        # Precompute associated Legendre polynomials (inverse normalisation)
        pct = legpoly(self.mmax, self.lmax, np.cos(theta), inverse=True)
        pct = torch.from_numpy(pct)

        self.register_buffer("pct", pct, persistent=False)

    def irfft_rings_reduced(self, x: Tensor) -> Tensor:
        """Inverse complex-to-real FFT on each ring of a reduced grid.

        Parameters
        ----------
        x : Tensor   shape [..., nlat, mmax]   (complex)

        Returns
        -------
        Tensor   shape [..., grid]
        """
        irfft = [
            torch.fft.irfft(x[..., t, :], nlon, norm="forward")
            for t, nlon in enumerate(self.lons_per_lat)
        ]
        return torch.cat(tensors=irfft, dim=-1)

    def irfft_rings_regular(self, x: Tensor) -> Tensor:
        """Inverse complex-to-real FFT on each ring of a regular grid.

        Parameters
        ----------
        x : Tensor   shape [..., nlat, mmax]   (complex)

        Returns
        -------
        Tensor   shape [..., grid]
        """
        return torch.fft.irfft(x, self.lons_per_lat[0], norm="forward").reshape(
            *x.shape[:-2], self.n_grid_points
        )

    def forward(self, x: Tensor) -> Tensor:
        """Inverse SHT: spectral coefficients → grid.

        Parameters
        ----------
        x : Tensor   shape [..., lmax, mmax]   (complex)

        Returns
        -------
        Tensor   shape [..., grid]
        """
        x = torch.view_as_real(x)

        rl = torch.einsum("...lm, mlk -> ...km", x[..., 0], self.pct.to(x.dtype))
        im = torch.einsum("...lm, mlk -> ...km", x[..., 1], self.pct.to(x.dtype))

        x = torch.stack((rl, im), -1).to(x.dtype)
        x = torch.view_as_complex(x)
        x = self.irfft_rings(x)
        return x
