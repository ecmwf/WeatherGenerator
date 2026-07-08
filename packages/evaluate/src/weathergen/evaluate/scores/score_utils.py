# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# Standard library
import logging
from typing import Any

# Third-party
import numpy as np
import xarray as xr
from omegaconf.listconfig import ListConfig

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


def to_list(obj: Any) -> list:
    """
    Convert given object to list if obj is not already a list. Sets are also transformed to a list.

    Parameters
    ----------
    obj : Any
        The object to transform into a list.
    Returns
    -------
    list
        A list containing the object, or the object itself if it was already a list.
    """
    if isinstance(obj, set | tuple | ListConfig):
        obj = list(obj)
    elif not isinstance(obj, list):
        obj = [obj]
    return obj


def calc_latitude_weights(
    data: xr.DataArray,
    min_value: float = 1e-3,
    max_value: float = 1.0,
    lat_coord_name: str | None = None,
) -> xr.DataArray:
    """
    Calculate latitude weights based on cosine of latitude.

    This function computes weights that account for the convergence of meridians
    towards the poles, giving less weight to high-latitude grid points.

    Parameters
    ----------
    data : xr.DataArray
        Data array with latitude coordinate
    min_value : float
        Minimum weight value (at poles). Default is 1e-3.
    max_value : float
        Maximum weight value (at equator). Default is 1.0.
    lat_coord_name : str | None
        Name of the latitude coordinate. If None, will search for standard
        latitude coordinate names ('lat', 'latitude', 'rlat', 'clat').

    Returns
    -------
    xr.DataArray
        Latitude weights as an xarray DataArray with the same dimensions as
        the latitude coordinate in the input data.

    Raises
    ------
    ValueError
        If no latitude coordinate is found in the data.
    """
    if lat_coord_name is None:
        lat_names = ["lat", "latitude", "rlat", "clat"]
        found_coords = [name for name in lat_names if name in data.coords]
        if not found_coords:
            raise ValueError(
                f"No latitude coordinate found. Please specify lat_coord_name. "
                f"Searched for: {lat_names}"
            )
        lat_coord_name = found_coords[0]

    lat_values = data.coords[lat_coord_name]
    lat_radians = np.deg2rad(lat_values)
    weights = (max_value - min_value) * np.cos(lat_radians) + min_value

    lat_dim = lat_values.dims[0] if len(lat_values.dims) > 0 else lat_coord_name
    return xr.DataArray(weights, coords={lat_coord_name: lat_values}, dims=[lat_dim])
