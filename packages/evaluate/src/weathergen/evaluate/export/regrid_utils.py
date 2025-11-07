import logging
import re
import numpy as np
import xarray as xr

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

"""
Enhanced functions to handle Gaussian grids when converting from Zarr to NetCDF.
"""

def detect_grid_type(input_data_array: xr.DataArray) -> str:
    """Detect whether data is on a regular lat/lon grid or Gaussian grid."""
    if "lat" not in input_data_array.coords or "lon" not in input_data_array.coords:
        return "unknown"

    lats = input_data_array.coords["lat"].values
    lons = input_data_array.coords["lon"].values

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

def find_pl(all_variables: list) -> tuple[dict[str, list[str]], list[int]]:
    """
    Find all the pressure levels for each variable using regex and returns a dictionary
    mapping variable names to their corresponding pressure levels.

    Parameters
    ----------
        all_variables : list of variable names with pressure levels (e.g.,'q_500','t_2m').

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
    for var in all_variables:
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


def reshape_dataset_adaptive(input_data_array: xr.DataArray) -> xr.Dataset:
    """
    Reshape dataset while preserving grid structure (regular or Gaussian).

    Parameters
    ----------
    input_data_array : xr.DataArray
        Input data with dimensions (ipoint, channel)

    Returns
    -------
    xr.Dataset
        Reshaped dataset appropriate for the grid type
    """
    grid_type = detect_grid_type(input_data_array)

    # Original logic
    var_dict, pl = find_pl(input_data_array.channel.values)
    data_vars = {}

    for new_var, old_vars in var_dict.items():
        if len(old_vars) > 1:
            data_vars[new_var] = xr.DataArray(
                input_data_array.sel(channel=old_vars).values,
                dims=["ipoint", "pressure_level"],
            )
        else:
            data_vars[new_var] = xr.DataArray(
                input_data_array.sel(channel=old_vars[0]).values,
                dims=["ipoint"],
            )

    reshaped_dataset = xr.Dataset(data_vars)
    reshaped_dataset = reshaped_dataset.assign_coords(
        ipoint=input_data_array.coords["ipoint"],
        pressure_level=pl,
    )

    if grid_type == "regular":
        # Use original reshape logic for regular grids
        # This is safe for regular grids
        reshaped_dataset = reshaped_dataset.set_index(ipoint=("valid_time", "lat", "lon")).unstack(
            "ipoint"
        )
    else:
        # Use new logic for Gaussian/unstructured grids
        reshaped_dataset = reshaped_dataset.set_index(ipoint2=("ipoint", "valid_time")).unstack(
            "ipoint2"
        )
        # rename ipoint to ncells
        reshaped_dataset = reshaped_dataset.rename_dims({"ipoint": "ncells"})
        reshaped_dataset = reshaped_dataset.rename_vars({"ipoint": "ncells"})

    return reshaped_dataset
