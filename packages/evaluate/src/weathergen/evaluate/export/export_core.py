import logging
from multiprocessing import Pool

import xarray as xr
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np

from weathergen.common.config import get_model_results
from weathergen.common.io import ZarrIO
from weathergen.evaluate.export.cf_utils import CF_ParserFactory
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

def get_fsteps(config, fname_zarr: str):
    fsteps = config.fsteps
    with ZarrIO(fname_zarr) as zio:
        zio_forecast_steps = sorted([int(step) for step in zio.forecast_steps])
    return zio_forecast_steps if fsteps is None else sorted([int(fstep) for fstep in fsteps])

def get_samples(config, fname_zarr: str):
    samples = config.samples
    with ZarrIO(fname_zarr) as zio:
        zio_samples = sorted([int(sample) for sample in zio.samples])
    samples =  (
                    zio_samples
                    if samples is None
                    else sorted([int(sample) for sample in samples if sample in samples])
    )
    return samples

def get_channels(config, stream: str, fname_zarr: str) -> list[str]:
    channels = config.channels
    with ZarrIO(fname_zarr) as zio:
        zio_forecast_steps = sorted([int(step) for step in zio.forecast_steps])
        dummy_out = zio.get_data(0, stream, zio_forecast_steps[0])
        all_channels = dummy_out.target.channels
        return all_channels if channels is None else channels
    
def get_grid_type(data_type, stream: str, fname_zarr: str) -> str:
    with ZarrIO(fname_zarr) as zio:
        zio_forecast_steps = sorted([int(step) for step in zio.forecast_steps])
        dummy_out = zio.get_data(0, stream, zio_forecast_steps[0])
        data = dummy_out.target if data_type == "target" else dummy_out.prediction
        return detect_grid_type(data.as_xarray().squeeze())


def get_ref_time(da: xr.Dataset, fstep_hours: np.timedelta64) -> np.datetime64:
    return 

#TODO: this will change after restructuring the lead time. 
def get_ref_times(fname_zarr, stream, samples, fstep_hours ) -> list[np.datetime64]:
        
    ref_times = []
    with ZarrIO(fname_zarr) as zio:
        zio_forecast_steps = sorted([int(step) for step in zio.forecast_steps])
        for sample in samples:
            data = zio.get_data(0, stream, zio_forecast_steps[0])
            data = data.target.as_xarray().squeeze()
            ref_time = data.valid_time.values[0] - fstep_hours * int(data.forecast_step.values)
            ref_times.append(ref_time)
    return ref_times

def export_model_outputs(data_type: str, config: OmegaConf) -> None:
    """
    Retrieve data from Zarr store and save one sample to each NetCDF file.
    Using multiprocessing to speed up data retrieval.

    Parameters
    ----------
    data_type: str
        Type of data to retrieve ('target' or 'prediction').
    config : OmegaConf
            Loaded config for cf_parser function.
    
    NOTE - config must contain the following parameters:
        run_id : str
            Run ID to identify the Zarr store.
        samples : list
            Sample to process
        stream : str
            Stream name to retrieve data for (e.g., 'ERA5').
        dtype : str
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
        output_dir : str
            Directory to save the NetCDF files.
        output_format : str
            Output file format (currently only 'netcdf' supported).
        
    """
    run_id = config.run_id
    samples = config.samples
    stream = config.stream
    channels = config.channels
    n_processes = config.n_processes
    epoch = config.epoch
    rank = config.rank
    output_dir = config.output_dir
    fstep_hours = np.timedelta64(config.fstep_hours, "h")
    output_format = config.output_format

    if data_type not in ["target", "prediction"]:
        raise ValueError(f"Invalid type: {data_type}. Must be 'target' or 'prediction'.")

    fname_zarr = get_model_results(run_id, epoch, rank)
    fsteps = get_fsteps(config, fname_zarr)
    samples = get_samples(config, fname_zarr)
    grid_type = get_grid_type(data_type, stream, fname_zarr)
    channels = get_channels(config, stream, fname_zarr)
    ref_times = get_ref_times(fname_zarr, stream, samples, fstep_hours)

    with Pool(processes=n_processes, maxtasksperchild=5) as pool:

        parser = CF_ParserFactory.get_parser(config, grid_type)

        for s_idx, sample in enumerate(tqdm(samples)):
            ref_time = ref_times[s_idx]
            #TODO: if sample file already exists, skip it. Add option to overwrite.
            da_fs = []
            step_tasks = [
                (sample, fstep, run_id, stream, data_type, epoch, rank) for fstep in fsteps
            ]
            for result in tqdm(
                pool.imap_unordered(get_data_worker, step_tasks, chunksize=1),
                total=len(step_tasks),
                desc=f"Processing {run_id} - stream: {stream} - sample: {sample}",
            ):
                if result is not None:
                    # Select only requested channels
                    result = result.as_xarray().squeeze()
                    if set(channels) != set(channels):
                        available_channels = result.channel.values
                        existing_channels = [ch for ch in channels if ch in available_channels]
                        if len(existing_channels) < len(channels):
                            _logger.info(
                                f"The following channels were not found: "
                                f"{list(set(channels) - set(existing_channels))}. Skipping them."
                            )
                        result = result.sel(channel=existing_channels)
                    # reshape result - use adaptive function to handle both regular and Gaussian
                    # grids
                    result = parser.reshape(result)
                    da_fs.append(result)

            _logger.info(f"Retrieved {len(da_fs)} forecast steps for type {data_type}.")
            _logger.info(
                f"Saving sample {sample} data to {output_format} format in {output_dir}."
            )
            
            da_fs = parser.concatenate(da_fs)
            da_fs = parser.assign_coords(da_fs, ref_time)
            da_fs = parser.add_attrs(da_fs)
            da_fs = parser.add_metadata(da_fs)
            
            parser.save(da_fs,data_type, ref_time)

        pool.terminate()
        pool.join()
