# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from dataclasses import dataclass
from typing import ClassVar

import xarray as xr

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


class RegionLibrary:
    """
    Predefined bounding boxes for known regions.
    """

    REGIONS: ClassVar[dict[str, tuple[float, float, float, float]]] = {
        "global": (-90.0, 90.0, -180.0, 180.0),
        "nhem": (0.0, 90.0, -180.0, 180.0),
        "shem": (-90.0, 0.0, -180.0, 180.0),
        "tropics": (-30.0, 30.0, -180.0, 180.0),
        "belgium": (49, 52, 2, 7),
    }


@dataclass(frozen=True)
class RegionBoundingBox:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def __post_init__(self):
        """Validate the bounding box coordinates."""
        self.validate()

    def validate(self):
        """Validate the bounding box coordinates."""
        if not (-90 <= self.lat_min <= 90 and -90 <= self.lat_max <= 90):
            raise ValueError(
                f"Latitude bounds must be between -90 and 90. Got: {self.lat_min}, {self.lat_max}"
            )
        if not (-180 <= self.lon_min <= 180 and -180 <= self.lon_max <= 180):
            raise ValueError(
                "Longitude bounds must be between -180 and 180. "
                + f"Got: {self.lon_min}, {self.lon_max}"
            )
        if self.lat_min >= self.lat_max:
            raise ValueError(
                f"Latitude minimum must be less than maximum. Got: {self.lat_min}, {self.lat_max}"
            )
        if self.lon_min >= self.lon_max:
            raise ValueError(
                f"Longitude minimum must be less than maximum. Got: {self.lon_min}, {self.lon_max}"
            )

    def contains(self, lat: float, lon: float) -> bool:
        """Check if a lat/lon point is within the bounding box."""
        return (self.lat_min <= lat <= self.lat_max) and (self.lon_min <= lon <= self.lon_max)

    def apply_mask(
        self,
        data: xr.Dataset | xr.DataArray,
        lat_name: str = "lat",
        lon_name: str = "lon",
        data_dim: str = "ipoint",
    ) -> xr.Dataset | xr.DataArray:
        """Filter Dataset or DataArray by spatial bounding box on 'ipoint' dimension.
        Parameters
        ----------
        data :
            The data to filter.
        lat_name:
            Name of the latitude coordinate in the data.
        lon_name:
            Name of the longitude coordinate in the data.
        data_dim:
            Name of the dimension that contains the lat/lon coordinates.

        Returns
        -------
        Filtered data with only points within the bounding box.
        """
        # lat/lon coordinates should be 1D and aligned with ipoint
        lat = data[lat_name]
        lon = data[lon_name]

        mask = (
            (lat >= self.lat_min)
            & (lat <= self.lat_max)
            & (lon >= self.lon_min)
            & (lon <= self.lon_max)
        )

        return data.sel({data_dim: mask})

    @classmethod
    def from_region_name(cls, region: str) -> "RegionBoundingBox":
        region = region.lower()
        try:
            return cls(*RegionLibrary.REGIONS[region])
        except KeyError as err:
            raise ValueError(
                f"Region '{region}' is not supported. "
                f"Available regions: {', '.join(RegionLibrary.REGIONS.keys())}"
            ) from err
