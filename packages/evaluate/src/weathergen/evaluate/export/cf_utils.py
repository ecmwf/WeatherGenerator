import logging

import numpy as np
import xarray as xr
from omegaconf import OmegaConf

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


def add_gaussian_grid_metadata(ds: xr.Dataset, grid_info: dict | None = None) -> xr.Dataset:
    """
    Add Gaussian grid metadata following CF conventions.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to add metadata to
    grid_info : dict, optional
        Dictionary with grid information:
        - 'N': Gaussian grid number (e.g., N320)
        - 'reduced': Whether it's a reduced Gaussian grid

    Returns
    -------
    xr.Dataset
        Dataset with added grid metadata
    """
    ds = ds.copy()
    # Add grid mapping information
    ds.attrs["grid_type"] = "gaussian"

    # If grid info provided, add it
    if grid_info:
        ds.attrs["gaussian_grid_number"] = grid_info.get("N", "unknown")
        ds.attrs["gaussian_grid_type"] = "reduced" if grid_info.get("reduced", False) else "regular"

    return ds


def add_conventions(stream: str, run_id: str, ds: xr.Dataset) -> xr.Dataset:
    """
    Add CF conventions to the dataset attributes.

    Parameters
    ----------
        stream : Stream name to include in the title attribute.
        run_id : Run ID to include in the title attribute.
        ds : Input xarray Dataset to add conventions to.
    Returns
    -------
        xarray Dataset with CF conventions added to attributes.
    """
    ds = ds.copy()
    ds.attrs["title"] = f"WeatherGenerator Output for {run_id} using stream {stream}"
    ds.attrs["institution"] = "WeatherGenerator Project"
    ds.attrs["source"] = "WeatherGenerator v0.0"
    ds.attrs["history"] = (
        "Created using the export_inference.py script on "
        + np.datetime_as_string(np.datetime64("now"), unit="s")
    )
    ds.attrs["Conventions"] = "CF-1.12"
    return ds


def cf_parser_gaussian_aware(config: OmegaConf, ds: xr.Dataset) -> xr.Dataset:
    """
    Modified CF parser that handles both regular and Gaussian grids.

    Parameters
    ----------
    config : OmegaConf
        Configuration for CF parsing
    ds : xr.Dataset
        Input dataset

    Returns
    -------
    xr.Dataset
        Parsed dataset with appropriate structure for grid type
    """
    # Detect if this is a Gaussian grid
    is_gaussian = "ncells" in ds.dims

    variables = {}
    mapping = config["variables"]

    # Handle dimensions based on grid type
    if is_gaussian:
        # For Gaussian grids, keep ncells and don't try to create lat/lon dimensions
        for var_name in ds.data_vars:
            if var_name in ["lat", "lon"]:
                continue

            variable = ds[var_name]

            if var_name not in mapping:
                # Variable not in mapping - skip or keep as-is
                variables[var_name] = variable
                continue

            dims = list(variable.dims)

            attributes = dict(
                standard_name=mapping[var_name].get("std", var_name),
                units=mapping[var_name].get("std_unit", "unknown"),
                coordinates="lat lon",  # Mark auxiliary coordinates
            )

            # Get mapped variable name or use original
            mapped_name = mapping[var_name].get("var", var_name)

            variables[mapped_name] = xr.DataArray(
                data=variable.values,
                dims=dims,
                coords={coord: ds.coords[coord] for coord in variable.coords if coord in ds.coords},
                attrs=attributes,
                name=mapped_name,
            )

        # Preserve lat/lon as coordinate variables with proper attributes
        if "lat" in ds.coords:
            ds.coords["lat"].attrs = {
                "standard_name": "latitude",
                "long_name": "latitude",
                "units": "degrees_north",
            }
        if "lon" in ds.coords:
            ds.coords["lon"].attrs = {
                "standard_name": "longitude",
                "long_name": "longitude",
                "units": "degrees_east",
            }

    else:
        # Original logic for regular grids
        ds_attributes = {}
        for dim_name, dim_dict in config["dimensions"].items():
            if dim_name == dim_dict["wg"]:
                dim_attributes = dict(standard_name=dim_dict.get("std", None))
                if dim_dict.get("std_unit", None) is not None:
                    dim_attributes["units"] = dim_dict["std_unit"]
                ds_attributes[dim_dict["wg"]] = dim_attributes
                continue

            if dim_name in ds.dims:
                ds = ds.rename_dims({dim_name: dim_dict["wg"]})

            dim_attributes = dict(standard_name=dim_dict.get("std", None))
            if "std_unit" in dim_dict and dim_dict["std_unit"] is not None:
                dim_attributes["units"] = dim_dict["std_unit"]
            ds_attributes[dim_dict["wg"]] = dim_attributes

        for var_name in ds.data_vars:
            dims = ["pressure", "valid_time", "latitude", "longitude"]
            if mapping[var_name]["level_type"] == "sfc":
                dims.remove("pressure")

            coordinates = {}
            for coord, new_name in config["coordinates"][mapping[var_name]["level_type"]].items():
                coordinates |= {
                    new_name: (
                        ds.coords[coord].dims,
                        ds.coords[coord].values,
                        ds_attributes[new_name],
                    )
                }

            variable = ds[var_name]
            attributes = dict(
                standard_name=mapping[var_name]["std"],
                units=mapping[var_name]["std_unit"],
            )

            variables[mapping[var_name]["var"]] = xr.DataArray(
                data=variable.values,
                dims=dims,
                coords={**coordinates, "valid_time": ds["valid_time"].values},
                attrs=attributes,
                name=mapping[var_name]["var"],
            )

    dataset = xr.merge(variables.values())
    dataset.attrs = ds.attrs

    return dataset
