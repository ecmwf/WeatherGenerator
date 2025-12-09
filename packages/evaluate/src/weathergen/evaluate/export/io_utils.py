import logging
from pathlib import Path

import numpy as np
import xarray as xr

from weathergen.common.config import get_model_results
from weathergen.common.io import ZarrIO

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


def output_filename(
    prefix: str,
    run_id: str,
    output_dir: str,
    output_format: str,
    forecast_ref_time: np.datetime64,
    regrid_degree: float,
) -> Path:
    """
    Generate output filename based on prefix (should refer to type e.g. pred/targ), run_id, sample
    index, output directory, format and forecast_ref_time.

    Parameters
    ----------
        prefix : Prefix for file name (e.g., 'pred' or 'targ').
        run_id :Run ID to include in the filename.
        output_dir : Directory to save the output file.
        output_format : Output file format (currently only 'netcdf' supported).
        forecast_ref_time : Forecast reference time to include in the filename.

    Returns
    -------
        Full path to the output file.
    """
    if output_format not in ["netcdf"]:
        raise ValueError(
            f"Unsupported output format: {output_format}, supported formates are ['netcdf']"
        )
    file_extension = "nc"
    frt = np.datetime_as_string(forecast_ref_time, unit="h")
    if regrid_degree is not None:
        run_id += f"_regular{regrid_degree, regrid_degree}"
    out_fname = Path(output_dir) / f"{prefix}_{frt}_{run_id}.{file_extension}"
    return out_fname


def get_data_worker(args: tuple) -> xr.DataArray:
    """
    Worker function to retrieve data for a single sample and forecast step.

    Parameters
    ----------
        args : Tuple containing (sample, fstep, run_id, stream, type).

    Returns
    -------
        xarray DataArray for the specified sample and forecast step.
    """
    sample, fstep, run_id, stream, dtype, epoch, rank = args
    fname_zarr = get_model_results(run_id, epoch, rank)
    with ZarrIO(fname_zarr) as zio:
        out = zio.get_data(sample, stream, fstep)
        if dtype == "target":
            data = out.target
        elif dtype == "prediction":
            data = out.prediction
    return data
