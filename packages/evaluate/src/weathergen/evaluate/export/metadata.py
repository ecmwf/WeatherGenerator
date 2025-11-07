import logging
import numpy as np
import xarray as xr

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