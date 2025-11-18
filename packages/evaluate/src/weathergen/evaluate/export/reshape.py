import logging
import re

import numpy as np
import xarray as xr

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

"""
Enhanced functions to handle Gaussian grids when converting from Zarr to NetCDF.
"""


def detect_grid_type(data: xr.DataArray) -> str:
    """
    Detect whether data is on a regular lat/lon grid or Gaussian grid.

    Parameters
    ----------
    data:
        input dataset.

    Returns
    -------
    str:
        String with the grid type.
        Supported options at the moment: "unknown", "regular", "gaussian"
    """
    if "lat" not in data.coords or "lon" not in data.coords:
        return "unknown"

    lats = data.coords["lat"].values
    lons = data.coords["lon"].values

    unique_lats = np.unique(lats)
    unique_lons = np.unique(lons)

    # Check if all (lat, lon) combinations exist (regular grid)
    if len(lats) == len(unique_lats) * len(unique_lons):
        lat_lon_pairs = set(zip(lats, lons, strict=False))
        expected_pairs = {(lat, lon) for lat in unique_lats for lon in unique_lons}
        if lat_lon_pairs == expected_pairs:
            return "regular"

    # Otherwise it's Gaussian (irregular spacing or reduced grid)
    return "gaussian"


def find_pl(vars: list) -> tuple[dict[str, list[str]], list[int]]:
    """
    Find all the pressure levels for each variable using regex and returns a dictionary
    mapping variable names to their corresponding pressure levels.

    Parameters
    ----------
        vars : list of variable names with pressure levels (e.g.,'q_500','t_2m').

    Returns
    -------
        A tuple containing:
        - var_dict: dict
            Dictionary mapping variable names to lists of their corresponding pressure levels.
        - pl: list of int
            List of unique pressure levels found in the variable names.
    """
    var_dict = {}
    pl = []
    for var in vars:
        match = re.search(r"^([a-zA-Z0-9_]+)_(\d+)$", var)
        if match:
            var_name = match.group(1)
            pressure_level = int(match.group(2))
            pl.append(pressure_level)
            var_dict.setdefault(var_name, []).append(var)
        else:
            var_dict.setdefault(var, []).append(var)
    pl = list(set(pl))
    return var_dict, pl

import pathlib
import xarray as xr
import numpy as np
import logging
from earthkit.regrid import interpolate
import os

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

def find_lat_lon_ordering(ds: xr.Dataset) -> list[int]:
    """
    Find all the the latitude and longitude ordering for the WeatherGenerator data
    Ordering from North West to South East.
    Returns the indices required to reorder the data.

    Parameters
    ----------
        ds : Input xarray Dataset containing the inference data,

    Returns
    -------
        indices: list of indices to reorder the data from original to lat/lon ordered.
    """
    x = ds['longitude'].values[:,0]
    y = ds['latitude'].values[:,0]
    tuples = list(zip(x, y))
    ordered_tuples = sorted(tuples, key=lambda t: (-t[1], t[0]))
    indices = [tuples.index(t) for t in ordered_tuples]
    return indices

def regrid_gaussian_ds(data: xr.Dataset, output_grid_type: str, degree: float, indices: list) -> xr.DataArray:
    """
    Regrid a single xarray Dataset from O96 grid to regular lat/lon grid.

    Parameters
    ----------
        indices: list of indices to reorder the data from original to lat/lon ordered.
        data : Input xarray DataArray containing the inference data on Gaussian grid.
        output_grid_type : Type of grid to regrid to (e.g., 'regular_ll').
        degree : Degree of the regular lat/lon grid (e.g., 2 for 2.0 degree grid)/96 for O96 grid

    Returns
    -------
        Regridded xarray DataArray.
    """
    #grid_type logic
    if output_grid_type == 'regular_ll':
        output_grid_type = [degree, degree]
        grid_shape = (180 // degree + 1, 360 // degree)
    else:
        raise ValueError(f'Unsupported grid_type: {output_grid_type}, supported types are ["regular_ll"]')
        # to be implemented:
        grid_type = grid_type + str(degree)
        grid_shape = None # ????? to be determined

    # reorder everything except ncells
    original_ncells = data['ncells']
    data = data.isel(ncells = indices)
    data['ncells'] = (original_ncells)

    # create empty xarray ds for regridded data
    values = np.empty((data.shape[0], grid_shape[0], grid_shape[1], data.shape[2]))
    for i in range(data.shape[0]):
        for j in range(data.shape[2]):
            data_reg_var = interpolate(
                data.values[i,:,j],
                {'grid': 'O96'},
                {'grid': output_grid_type}
                )
            values[i,:,:,j] = data_reg_var
    dims = list(data.dims)
    pos = dims.index('ncells')
    dims[pos:pos+1] = ['latitude', 'longitude']
    dims = tuple(dims)
    regrid_data = xr.DataArray(
        data = values,
        dims = dims,
        coords = {
            'valid_time': data['valid_time'].values,
            'latitude': np.linspace(-90, 90, grid_shape[0]),
            'longitude': np.linspace(0, 360 - degree, grid_shape[1]),
        },
        attrs = data.attrs,
        name = data.name
    )
    return regrid_data