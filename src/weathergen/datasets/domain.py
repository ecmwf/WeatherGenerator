# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Regional HEALPix subdomain support (domain cropping).

The rest of the code base addresses healpix cells by their *global nested index*
in [0, 12 * 4**hl). When a regional bounding box is configured, only a subset of
those cells is active, and every per-cell array (masks, token lists, pe_global,
neighbour tables, ...) has length `num_active` instead of 12 * 4**hl.

This module is the single source of truth for that mapping:

    global nested index  --(to_compact)-->   compact index in [0, num_active)
    compact index        --(active_cells)--> global nested index

Design notes
------------
* `Domain` is a plain, picklable, torch-free object holding only numpy arrays and
  scalars, so it survives being sent to dataloader worker processes.

* The global domain is a first-class case: `Domain.global_(hl)` returns a domain
  where `to_compact` is the identity. With no `domain:` block in the config the
  behaviour is bit-identical to the code without cropping.

* Cell selection is by cell *centre*. A cell whose centre lies just outside the box
  but which still contains points inside the box is dropped (and so are those
  points). Use `pad_rings` to grow the domain by N rings of neighbours if a margin
  is wanted.

* Longitude convention: the tokenizer (`theta_phi_to_standard_coords`) maps a
  geographic longitude `lon` to the healpix azimuth `phi = (lon + 180) / 360 * 2 pi`.
  Everything here uses the same convention, so the cells selected by the bbox are
  exactly the cells the tokenizer assigns data to.

Config
------
.. code-block:: yaml

    domain:
      enabled: true      # optional, default true when the block is present
      lon_min: -10.0     # degrees in [-180, 180]; lon_min > lon_max crosses 180
      lon_max: 35.0
      lat_min: 50.0
      lat_max: 75.0
      pad_rings: 1       # optional, default 0
"""

import logging
import warnings

import astropy_healpix as hp
import numpy as np
from astropy_healpix.healpy import ang2pix, pix2ang
from numpy.typing import NDArray

_logger = logging.getLogger(__name__)

# Domains are built from the config in several places (sampler, model, encoder);
# cache them so the construction (and its log line) happens once per process.
_DOMAIN_CACHE: dict[tuple, "Domain"] = {}


def _lonlat_to_thetaphi(lats_deg: NDArray, lons_deg: NDArray) -> tuple[NDArray, NDArray]:
    """Same convention as tokenizer_utils.theta_phi_to_standard_coords."""
    thetas = ((90.0 - lats_deg) / 180.0) * np.pi
    phis = ((lons_deg + 180.0) / 360.0) * 2.0 * np.pi
    return thetas, phis


def _cells_centres_lonlat(healpix_level: int, cells: NDArray) -> tuple[NDArray, NDArray]:
    """(lons, lats) in degrees of the given global cells, inverting the convention above."""
    theta, phi = pix2ang(2**healpix_level, cells, nest=True)
    lats_deg = 90.0 - np.degrees(theta)
    lons_deg = np.degrees(phi) - 180.0
    return lons_deg, lats_deg


class Domain:
    """
    A set of active healpix cells at a fixed healpix level.

    Attributes
    ----------
    healpix_level : int
        The healpix level (cf.healpix_level).
    active_cells : NDArray[np.int64]
        Sorted global nested indices of the active cells, shape (num_active,).
        `active_cells[c]` is the global index of compact index `c`.
    to_compact : NDArray[np.int64]
        Lookup of length 12 * 4**hl. `to_compact[g]` is the compact index of
        global index `g`, or -1 if `g` is outside the domain.
    is_global : bool
        True if the domain covers the whole sphere (then to_compact is the identity).
    bbox : tuple | None
        (lon_min, lon_max, lat_min, lat_max) in degrees, or None if global.
    pad_rings : int
        Number of neighbour rings added around the cells selected by the bbox.
    """

    def __init__(
        self,
        healpix_level: int,
        active_cells: NDArray[np.int64],
        is_global: bool = False,
        bbox: tuple | None = None,
        pad_rings: int = 0,
    ) -> None:
        num_total = 12 * 4**healpix_level

        self.healpix_level = healpix_level
        self.active_cells = np.asarray(active_cells, dtype=np.int64)
        self.is_global = is_global
        self.bbox = bbox
        self.pad_rings = pad_rings

        assert self.active_cells.ndim == 1, "active_cells must be 1D"
        assert len(self.active_cells) > 0, (
            "Domain is empty: no healpix cells fall inside the configured bounding box. "
            "Either the box is too small for this healpix level, or lat/lon are swapped."
        )
        assert np.all(np.diff(self.active_cells) > 0), "active_cells must be sorted and unique"
        assert self.active_cells[0] >= 0 and self.active_cells[-1] < num_total, (
            "active_cells out of range for this healpix level"
        )

        self.to_compact = np.full(num_total, -1, dtype=np.int64)
        self.to_compact[self.active_cells] = np.arange(len(self.active_cells), dtype=np.int64)

    def __len__(self) -> int:
        """Number of active cells. Replaces `12 * 4**healpix_level` everywhere."""
        return len(self.active_cells)

    @property
    def num_total_cells(self) -> int:
        """Number of cells the full sphere has at this level."""
        return 12 * 4**self.healpix_level

    # -----------------------------------------------------------------------
    # constructors
    # -----------------------------------------------------------------------

    @classmethod
    def global_(cls, healpix_level: int) -> "Domain":
        """The whole sphere. to_compact is the identity; behaviour is unchanged."""
        num_total = 12 * 4**healpix_level
        return cls(healpix_level, np.arange(num_total, dtype=np.int64), is_global=True)

    @classmethod
    def from_bbox(
        cls,
        healpix_level: int,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        pad_rings: int = 0,
    ) -> "Domain":
        """
        Build a domain from a lon/lat bounding box in degrees.

        Parameters
        ----------
        lon_min, lon_max :
            Longitude bounds in degrees, in [-180, 180]. If lon_min > lon_max the box
            crosses the antimeridian, e.g. lon_min=170, lon_max=-170.
        lat_min, lat_max :
            Latitude bounds in degrees, in [-90, 90], with lat_min < lat_max.
        pad_rings :
            Grow the selected set by this many rings of healpix neighbours.
        """
        assert lat_min < lat_max, f"lat_min must be < lat_max, got {lat_min}, {lat_max}"
        assert lat_min >= -90.0 and lat_max <= 90.0, "latitudes must be in [-90, 90]"
        assert -180.0 <= lon_min <= 180.0 and -180.0 <= lon_max <= 180.0, (
            "longitudes must be in [-180, 180]"
        )
        assert pad_rings >= 0, "pad_rings must be >= 0"

        nside = 2**healpix_level
        num_total = 12 * 4**healpix_level

        lons_deg, lats_deg = _cells_centres_lonlat(
            healpix_level, np.arange(num_total, dtype=np.int64)
        )

        in_lat = (lats_deg >= lat_min) & (lats_deg <= lat_max)
        if lon_min <= lon_max:
            in_lon = (lons_deg >= lon_min) & (lons_deg <= lon_max)
        else:
            in_lon = (lons_deg >= lon_min) | (lons_deg <= lon_max)

        selected = np.flatnonzero(in_lat & in_lon).astype(np.int64)

        assert len(selected) > 0, (
            f"No healpix cells at level {healpix_level} fall inside bbox "
            f"lon=[{lon_min}, {lon_max}] lat=[{lat_min}, {lat_max}]. "
            "The box is likely smaller than a single cell, or lat/lon are swapped."
        )

        for _ in range(pad_rings):
            selected = _grow_one_ring(selected, nside)

        _logger.info(
            "Domain: healpix level %d, bbox lon=[%g, %g] lat=[%g, %g], pad_rings=%d "
            "-> %d of %d cells active (%.2f%% of the sphere)",
            healpix_level,
            lon_min,
            lon_max,
            lat_min,
            lat_max,
            pad_rings,
            len(selected),
            num_total,
            100.0 * len(selected) / num_total,
        )

        return cls(
            healpix_level,
            selected,
            is_global=(len(selected) == num_total),
            bbox=(lon_min, lon_max, lat_min, lat_max),
            pad_rings=pad_rings,
        )

    @classmethod
    def from_config(cls, cf) -> "Domain":
        """
        Build the domain from the optional top-level `domain` config block.

        If the block is absent or `domain.enabled` is false, returns the global
        domain, i.e. exactly the behaviour without cropping.
        """
        dom_cfg = cf.get("domain", None)
        if dom_cfg is None or not dom_cfg.get("enabled", True):
            key = (cf.healpix_level,)
        else:
            key = (
                cf.healpix_level,
                float(dom_cfg["lon_min"]),
                float(dom_cfg["lon_max"]),
                float(dom_cfg["lat_min"]),
                float(dom_cfg["lat_max"]),
                int(dom_cfg.get("pad_rings", 0)),
            )

        domain = _DOMAIN_CACHE.get(key)
        if domain is None:
            domain = cls.global_(key[0]) if len(key) == 1 else cls.from_bbox(*key)
            _DOMAIN_CACHE[key] = domain
        return domain

    # -----------------------------------------------------------------------
    # index mapping
    # -----------------------------------------------------------------------

    def remap(self, global_idxs: NDArray) -> NDArray[np.int64]:
        """
        Map global nested indices to compact indices. Out-of-domain -> -1.

        Hot path: called once per stream per time window on the full point set.
        """
        if self.is_global:
            return global_idxs.astype(np.int64, copy=False)
        return self.to_compact[global_idxs]

    def inside_mask(self, global_idxs: NDArray) -> NDArray[np.bool_]:
        """Boolean mask of which global cell indices are in the domain."""
        if self.is_global:
            return np.ones(len(global_idxs), dtype=bool)
        return self.to_compact[global_idxs] >= 0

    def point_mask(self, lats: NDArray, lons: NDArray) -> NDArray[np.bool_]:
        """
        Boolean mask of which points (lat/lon in degrees) fall in an active cell.

        Used by the data readers to drop points before normalisation/tokenisation.
        It uses the same cell assignment as the tokenizer (hpy_cell_splits), so it
        keeps exactly the points the tokenizer would keep; the tokenizer still
        remains the authoritative filter for readers that do not pre-filter.
        """
        lats = np.asarray(lats)
        lons = np.asarray(lons)
        if self.is_global:
            return np.ones(len(lats), dtype=bool)

        mask = np.zeros(len(lats), dtype=bool)
        finite = np.isfinite(lats) & np.isfinite(lons)
        if finite.any():
            thetas, phis = _lonlat_to_thetaphi(
                lats[finite].astype(np.float64), lons[finite].astype(np.float64)
            )
            cells = ang2pix(2**self.healpix_level, thetas, phis, nest=True)
            mask[finite] = self.inside_mask(cells)
        return mask

    # -----------------------------------------------------------------------
    # geometry, remapped to compact indexing
    # -----------------------------------------------------------------------

    def neighbours_compact(self) -> NDArray[np.int64]:
        """
        The 8-neighbour table of the active cells, in compact indexing.

        Returns shape (num_active, 8). Neighbours that are missing in healpix (the -1
        returned at base-pixel corners) OR that fall outside the domain are replaced
        by the cell itself. This is the same "replace by self" convention the model
        already used for missing neighbours, so a domain-edge cell behaves like a
        healpix-corner cell.
        """
        nside = 2**self.healpix_level
        with warnings.catch_warnings(action="ignore"):
            nb = hp.neighbours(self.active_cells, nside, order="nested").transpose()

        nb_compact = np.full(nb.shape, -1, dtype=np.int64)
        valid = nb >= 0
        nb_compact[valid] = self.to_compact[nb[valid]]

        self_idx = np.arange(len(self), dtype=np.int64)[:, None]
        return np.where(nb_compact < 0, self_idx, nb_compact)

    def centres_lonlat(self) -> tuple[NDArray, NDArray]:
        """(lons, lats) in degrees of the active cell centres, compact order."""
        return _cells_centres_lonlat(self.healpix_level, self.active_cells)

    def __repr__(self) -> str:
        if self.is_global:
            return f"Domain(global, hl={self.healpix_level}, num_cells={len(self)})"
        return (
            f"Domain(bbox={self.bbox}, pad_rings={self.pad_rings}, hl={self.healpix_level}, "
            f"num_cells={len(self)}/{self.num_total_cells})"
        )


def _grow_one_ring(cells: NDArray[np.int64], nside: int) -> NDArray[np.int64]:
    """Add one ring of healpix neighbours around the given cell set."""
    with warnings.catch_warnings(action="ignore"):
        nb = hp.neighbours(cells, nside, order="nested")
    nb = nb[nb >= 0]
    return np.unique(np.concatenate([cells, nb.astype(np.int64)]))
