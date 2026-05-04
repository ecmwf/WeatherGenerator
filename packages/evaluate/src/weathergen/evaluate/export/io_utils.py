import logging

import xarray as xr

from weathergen.common.config import get_model_results
from weathergen.common.io import zarrio_reader

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


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
    with zarrio_reader(fname_zarr) as zio:
        out = zio.get_data(sample, stream, fstep)
        if dtype == "target":
            data = out.target
        elif dtype == "prediction":
            data = out.prediction
    return data
