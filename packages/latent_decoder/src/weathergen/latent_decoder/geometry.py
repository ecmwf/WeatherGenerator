"""HEALPix level-5 geometry of the latent tokens: the cell of a point, its neighbours, and where
in the cell the point lies.

Token c is the cell WeatherGenerator's tokenizer calls c. The tokenizer computes HEALPix nested
indices with phi = lon + 180 deg (datasets/tokenizer_utils.py), so token c covers the standard
HEALPix nested pixel c rotated by 180 deg in longitude. The `coords` the latent writer stores use
the standard pixel centres and are therefore 180 deg off. The rotation maps the HEALPix grid
onto itself, so neighbour indices are the same in both conventions.
"""

from functools import cache

import astropy.units as u
import astropy_healpix as hp
import numpy as np
import numpy.typing as npt

LEVEL = 5
NSIDE = 2**LEVEL
N_CELLS = 12 * NSIDE**2
LON_SHIFT = 180.0  # tokenizer longitude minus geographic longitude, degrees
# cells are ~1.8 degrees across; this scales a point's offset from its cell centre to ~1
OFFSET_SCALE = 1 / np.deg2rad(1.8)


def cell_of(lat, lon) -> npt.NDArray:
    lon = np.asarray(lon, dtype=np.float64) + LON_SHIFT
    return np.asarray(
        hp.lonlat_to_healpix(lon * u.deg, np.asarray(lat) * u.deg, NSIDE, order="nested"),
        dtype=np.int64,
    )


@cache
def cell_centres() -> tuple[npt.NDArray, npt.NDArray]:
    """Geographic (lat, lon) in degrees of each token's cell centre, lon in [-180, 180)."""
    lon, lat = hp.healpix_to_lonlat(np.arange(N_CELLS), NSIDE, order="nested")
    return lat.to_value(u.deg), (lon.to_value(u.deg) - LON_SHIFT + 180.0) % 360.0 - 180.0


@cache
def neighbours() -> npt.NDArray:
    """[cell, 9]: the cell itself, then its 8 neighbours. The 24 cells that have only 7
    neighbours repeat themselves in the missing slot."""
    cells = np.arange(N_CELLS)
    with np.errstate(invalid="ignore"):
        nb = hp.neighbours(cells, NSIDE, order="nested").T
    nb = np.where(nb < 0, cells[:, None], nb)
    return np.concatenate([cells[:, None], nb], axis=1).astype(np.int64)


def unit_vectors(lat, lon) -> npt.NDArray:
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


@cache
def _centre_vectors() -> npt.NDArray:
    return unit_vectors(*cell_centres())


def position_features(lat, lon, cells) -> npt.NDArray:
    """[n, 6]: the point's unit vector, and its offset from the centre of its cell."""
    xyz = unit_vectors(lat, lon)
    return np.concatenate([xyz, (xyz - _centre_vectors()[cells]) * OFFSET_SCALE], axis=1).astype(
        np.float32
    )
