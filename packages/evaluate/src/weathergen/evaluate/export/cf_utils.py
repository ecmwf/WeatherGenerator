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
    CF parser that handles both regular and Gaussian grids.

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
    # order important here
    if is_gaussian:
        dims_list = ["pressure", "ncells", "valid_time"]
    else:
        dims_list = [
            "pressure",
            "valid_time",
            "latitude",
            "longitude",
        ]
    # Start a new xarray dataset from scratch, it's easier than deleting / renaming (I tried!).
    variables = {}
    mapping = config["variables"]

    ds_attributes = {}
    for dim_name, dim_dict in config["dimensions"].items():
        # clear dimensions if key and dim_dict['wg'] are the same
        if dim_name == dim_dict["wg"]:
            dim_attributes = dict(
                standard_name=dim_dict.get("std", None),
            )
            if dim_dict.get("std_unit", None) is not None:
                dim_attributes["units"] = dim_dict["std_unit"]
            ds_attributes[dim_dict["wg"]] = dim_attributes
            continue
        if dim_name in ds.dims:
            ds = ds.rename_dims({dim_name: dim_dict["wg"]})
        dim_attributes = dict(
            standard_name=dim_dict.get("std", None),
        )
        if "std_unit" in dim_dict and dim_dict["std_unit"] is not None:
            dim_attributes["units"] = dim_dict["std_unit"]
        ds_attributes[dim_dict["wg"]] = dim_attributes
    for var_name in ds:
        dims = dims_list.copy()
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
            long_name=mapping[var_name]["long"],
        )
        if is_gaussian:
            # adding auxiliary coordinates
            attributes["coordinates"] = "latitude longitude"
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