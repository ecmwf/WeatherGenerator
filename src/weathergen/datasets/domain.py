"""Shared compact HEALPix domains, indexed by global NESTED cell ids."""

import warnings
from functools import lru_cache

import astropy_healpix as hp
import numpy as np

from weathergen.common.config import get_healpix_levels, validate_domains


class Domain:
    def __init__(self, healpix_level, bbox=None):
        self.healpix_level = healpix_level
        cells = np.arange(12 * 4**healpix_level, dtype=np.int64)
        if bbox is not None:
            lon_min, lon_max, lat_min, lat_max, pad_rings = bbox
            lon, lat = hp.healpix_to_lonlat(cells, 2**healpix_level, order="nested")
            lon, lat = lon.to_value("deg") - 180.0, lat.to_value("deg")
            in_lon = (
                (lon >= lon_min) & (lon <= lon_max)
                if lon_min < lon_max
                else (lon >= lon_min) | (lon <= lon_max)
            )
            cells = cells[in_lon & (lat >= lat_min) & (lat <= lat_max)]
            if not len(cells):
                raise ValueError(
                    f"Domain at HEALPix level {healpix_level} contains no cell centres."
                )
            for _ in range(pad_rings):
                with warnings.catch_warnings(action="ignore"):
                    neighbours = hp.neighbours(cells, 2**healpix_level, order="nested")
                cells = np.union1d(cells, neighbours[neighbours >= 0])
        self.active_cells = cells
        self.to_compact = np.full(12 * 4**healpix_level, -1, dtype=np.int64)
        self.to_compact[cells] = np.arange(len(cells), dtype=np.int64)
        self.active_cells.flags.writeable = False
        self.to_compact.flags.writeable = False

    @classmethod
    def from_config(cls, cf, level):
        validate_domains(cf)
        if type(level) is not int or level not in get_healpix_levels(cf):
            raise ValueError(f"Domain level {level} is not an existing stream level.")
        bbox = cf.get("domains", {}).get(level)
        settings = (
            None
            if bbox is None
            else (
                bbox["lon_min"],
                bbox["lon_max"],
                bbox["lat_min"],
                bbox["lat_max"],
                bbox.get("pad_rings", 0),
            )
        )
        return _domain(level, settings)

    def __len__(self):
        return len(self.active_cells)

    def neighbours_compact(self):
        """Return N x 8 compact neighbours; missing and outside neighbours refer to self."""
        with warnings.catch_warnings(action="ignore"):
            neighbours = hp.neighbours(self.active_cells, 2**self.healpix_level, order="nested").T
        compact = self.to_compact[np.maximum(neighbours, 0)]
        return np.where((neighbours >= 0) & (compact >= 0), compact, np.arange(len(self))[:, None])

    @lru_cache(maxsize=64)
    def centres_lonlat(self):
        """Return longitude, latitude degrees in the tokenizer's phi=(lon+180) convention."""
        lon, lat = hp.healpix_to_lonlat(self.active_cells, 2**self.healpix_level, order="nested")
        return lon.to_value("deg") - 180.0, lat.to_value("deg")


@lru_cache(maxsize=64)
def _domain(level, bbox):
    return Domain(level, bbox)
