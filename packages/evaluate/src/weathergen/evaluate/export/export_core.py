import logging
from multiprocessing import Pool

import numpy as np
import xarray as xr
from omegaconf import OmegaConf
from tqdm import tqdm

from weathergen.common.config import get_model_results
from weathergen.common.io import ZarrIO
from weathergen.evaluate.export.parser_factory import CfParserFactory
from weathergen.evaluate.export.reshape import detect_grid_type

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
    with ZarrIO(fname_zarr) as zio:
        out = zio.get_data(sample, stream, fstep)
        if dtype == "target":
            data = out.target
        elif dtype == "prediction":
            data = out.prediction
    return data


def get_fsteps(fsteps, fname_zarr: str):
    """
    Retrieve available forecast steps from the Zarr store and filter
    based on requested forecast steps.

    Parameters
    ----------
        fsteps : list
            List of requested forecast steps.
            If None, retrieves all available forecast steps.
        fname_zarr : str
            Path to the Zarr store.
    Returns
    -------
        list[str]
            List of forecast steps to be used for data retrieval.
    """
    with ZarrIO(fname_zarr) as zio:
        zio_forecast_steps = sorted([int(step) for step in zio.forecast_steps])
    return zio_forecast_steps if fsteps is None else sorted([int(fstep) for fstep in fsteps])


def get_samples(samples, fname_zarr: str):
    """
    Retrieve available samples from the Zarr store
    and filter based on requested samples.
    Parameters
    ----------
        samples : list
            List of requested samples. If None, retrieves all available samples.
        fname_zarr : str
            Path to the Zarr store.
    Returns
    -------
        list[str]
            List of samples to be used for data retrieval.
    """
    with ZarrIO(fname_zarr) as zio:
        zio_samples = sorted([int(sample) for sample in zio.samples])
    samples = (
        zio_samples
        if samples is None
        else sorted([int(sample) for sample in samples if sample in samples])
    )
    return samples


def get_channels(channels, stream: str, fname_zarr: str) -> list[str]:
    """
    Retrieve available channels from the Zarr store and filter based on requested channels.
    Parameters
    ----------
        channels : list
            List of requested channels. If None, retrieves all available channels.
        stream : str
            Stream name to retrieve data for (e.g., 'ERA5').
        fname_zarr : str
            Path to the Zarr store.
    Returns
    -------
        list[str]
            List of channels to be used for data retrieval.
    """
    with ZarrIO(fname_zarr) as zio:
        zio_forecast_steps = sorted([int(step) for step in zio.forecast_steps])
        dummy_out = zio.get_data(0, stream, zio_forecast_steps[0])
        all_channels = dummy_out.target.channels

        if channels is not None:
            existing_channels = set(all_channels) & set(channels)
            if existing_channels != set(channels):
                missing_channels = set(channels) - set(existing_channels)
                _logger.warning(
                    "The following requested channels are"
                    f"not available in the data and will be skipped: {missing_channels}"
                )
        return all_channels if channels is None else list(existing_channels)


def get_grid_type(data_type, stream: str, fname_zarr: str) -> str:
    """
    Determine the grid type of the data (regular or gaussian).
    Parameters
    ----------
        data_type : str
            Type of data to retrieve ('target' or 'prediction').
        stream : str
            Stream name to retrieve data for (e.g., 'ERA5').
        fname_zarr : str
            Path to the Zarr store.
    Returns
    -------
        str
            Grid type ('regular' or 'gaussian').
    """
    with ZarrIO(fname_zarr) as zio:
        zio_forecast_steps = sorted([int(step) for step in zio.forecast_steps])
        dummy_out = zio.get_data(0, stream, zio_forecast_steps[0])
        data = dummy_out.target if data_type == "target" else dummy_out.prediction
        return detect_grid_type(data.as_xarray().squeeze())


# TODO: this will change after restructuring the lead time.
def get_ref_times(fname_zarr, stream, samples, fstep_hours) -> list[np.datetime64]:
    """
    Retrieve reference times for the specified samples from the Zarr store.
    Parameters
    ----------
        fname_zarr : str
            Path to the Zarr store.
        stream : str
            Stream name to retrieve data for (e.g., 'ERA5').
        samples : list
            List of samples to process.
        fstep_hours : np.timedelta64
            Time difference between forecast steps in hours.
    Returns
    -------
        list[np.datetime64]
            List of reference times corresponding to the samples.
    """
    ref_times = []
    with ZarrIO(fname_zarr) as zio:
        zio_forecast_steps = sorted([int(step) for step in zio.forecast_steps])
        for sample in samples:
            data = zio.get_data(sample, stream, zio_forecast_steps[0])
            data = data.target.as_xarray().squeeze()
            ref_time = data.valid_time.values[0] - fstep_hours * int(data.forecast_step.values)
            ref_times.append(ref_time)
    return ref_times


def export_model_outputs(data_type: str, config: OmegaConf, **kwargs) -> None:
    """
    Retrieve data from Zarr store and save one sample to each NetCDF file.
    Using multiprocessing to speed up data retrieval.

    Parameters
    ----------
    data_type: str
        Type of data to retrieve ('target' or 'prediction').
    config : OmegaConf
            Loaded config for cf_parser function.

    kwargs:
        Additional keyword arguments for the parser.

    NOTE: it contains the following parameters:
        run_id : str
            Run ID to identify the Zarr store.
        samples : list
            Sample to process
        stream : str
            Stream name to retrieve data for (e.g., 'ERA5').
        data_type : str
            Type of data to retrieve ('target' or 'prediction').
        fsteps : list
            List of forecast steps to retrieve. If None, retrieves all available forecast steps.
        channels : list
            List of channels to retrieve. If None, retrieves all available channels.
        n_processes : list
            Number of parallel processes to use for data retrieval.
        ecpoch : int
            Epoch number to identify the Zarr store.
        rank : int
            Rank number to identify the Zarr store.
        regrid_degree : float
            If specified, regrid the data to a regular lat/lon grid with the given degree
        output_dir : str
            Directory to save the NetCDF files.
        output_format : str
            Output file format (currently only 'netcdf' supported).

    """
    kwargs = OmegaConf.create(kwargs)

    run_id = kwargs.run_id
    samples = kwargs.samples
    fsteps = kwargs.fsteps
    stream = kwargs.stream
    channels = kwargs.channels
    n_processes = kwargs.n_processes
    epoch = kwargs.epoch
    rank = kwargs.rank
    fstep_hours = np.timedelta64(kwargs.fstep_hours, "h")

    if data_type not in ["target", "prediction"]:
        raise ValueError(f"Invalid type: {data_type}. Must be 'target' or 'prediction'.")

    fname_zarr = get_model_results(run_id, epoch, rank)
    fsteps = get_fsteps(fsteps, fname_zarr)
    samples = get_samples(samples, fname_zarr)
    grid_type = get_grid_type(data_type, stream, fname_zarr)
    channels = get_channels(channels, stream, fname_zarr)
    ref_times = get_ref_times(fname_zarr, stream, samples, fstep_hours)

    kwargs["grid_type"] = grid_type
    kwargs["channels"] = channels
    kwargs["data_type"] = data_type

    with Pool(processes=n_processes, maxtasksperchild=5) as pool:
        parser = CfParserFactory.get_parser(config=config, **kwargs)

        for s_idx, sample in enumerate(tqdm(samples)):
            ref_time = ref_times[s_idx]

            step_tasks = [
                (sample, fstep, run_id, stream, data_type, epoch, rank) for fstep in fsteps
            ]

            results_iterator = pool.imap_unordered(get_data_worker, step_tasks, chunksize=1)

            parser.process_sample(
                results_iterator,
                ref_time=ref_time,
            )

        pool.terminate()
        pool.join()
