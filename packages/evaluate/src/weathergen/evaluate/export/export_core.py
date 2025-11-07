import logging
from multiprocessing import Pool

import xarray as xr
from omegaconf import OmegaConf
from tqdm import tqdm

from weathergen.common.config import get_model_results
from weathergen.common.io import ZarrIO
from weathergen.evaluate.export.cf_utils import (
    add_conventions,
    add_gaussian_grid_metadata,
    cf_parser_gaussian_aware,
)
from weathergen.evaluate.export.io_utils import get_data_worker, output_filename
from weathergen.evaluate.export.reshape import reshape_dataset_adaptive

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


def export_model_outputs(
    run_id: str,
    samples: list,
    stream: str,
    dtype: str,
    fsteps: list,
    channels: list,
    fstep_hours: int,
    n_processes: list,
    epoch: int,
    rank: int,
    output_dir: str,
    output_format: str,
    config: OmegaConf,
) -> None:
    """
    Retrieve data from Zarr store and save one sample to each NetCDF file.
    Using multiprocessing to speed up data retrieval.

    Parameters
    ----------
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
        config : OmegaConf
            Loaded config for cf_parser function.
    """
    if dtype not in ["target", "prediction"]:
        raise ValueError(f"Invalid type: {dtype}. Must be 'target' or 'prediction'.")

    fname_zarr = get_model_results(run_id, epoch, rank)
    with ZarrIO(fname_zarr) as zio:
        zio_forecast_steps = sorted([int(step) for step in zio.forecast_steps])
        zio_samples = sorted([int(sample) for sample in zio.samples])
        dummy_out = zio.get_data(0, stream, zio_forecast_steps[0])
        all_channels = dummy_out.target.channels
        channels = all_channels if channels is None else channels

    fsteps = zio_forecast_steps if fsteps is None else sorted([int(fstep) for fstep in fsteps])

    samples = (
        zio_samples
        if samples is None
        else sorted([int(sample) for sample in samples if sample in samples])
    )
    with Pool(processes=n_processes, maxtasksperchild=5) as pool:
        for sample_idx in tqdm(samples):
            da_fs = []
            step_tasks = [
                (sample_idx, fstep, run_id, stream, dtype, epoch, rank) for fstep in fsteps
            ]
            for result in tqdm(
                pool.imap_unordered(get_data_worker, step_tasks, chunksize=1),
                total=len(step_tasks),
                desc=f"Processing {run_id} - stream: {stream} - sample: {sample_idx}",
            ):
                if result is not None:
                    # Select only requested channels
                    result = result.as_xarray().squeeze()
                    if set(channels) != set(all_channels):
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
                    result = reshape_dataset_adaptive(result)
                    da_fs.append(result)

            _logger.info(f"Retrieved {len(da_fs)} forecast steps for type {dtype}.")
            _logger.info(
                f"Saving sample {sample_idx} data to {output_format} format in {output_dir}."
            )

            save_sample_to_netcdf(
                str(dtype)[:4],
                da_fs,
                fstep_hours,
                run_id,
                output_dir,
                output_format,
                config,
            )
        pool.terminate()
        pool.join()


def save_sample_to_netcdf(
    type_str,
    array_list,
    fstep_hours,
    run_id,
    output_dir,
    output_format,
    config,
) -> None:
    """
    Uses list of pred/target xarray DataArrays to save one sample to a NetCDF file.

    Parameters
    ----------
    type_str : str
        Type of data ('pred' or 'targ') to include in the filename.
    dict_sample_all_steps : dict
        Dictionary where keys is sample index and values is a list of xarray DataArrays
        for all the forecast steps
    fstep_hours : np.timedelta64
        Time difference between forecast steps (e.g., 6 hours).
    run_id : str
        Run ID to include in the filename.
    output_dir : str
        Directory to save the NetCDF files.
    output_format : str
        Output file format (currently only 'netcdf' supported).
    config : OmegaConf
        Loaded config for cf_parser function.
    """
    # find forecast_ref_time
    frt = array_list[0].valid_time.values[0] - fstep_hours * int(array_list[0].forecast_step.values)
    out_fname = output_filename(type_str, run_id, output_dir, output_format, frt)
    # check if file already exists
    if out_fname.exists():
        _logger.info(f"File {out_fname} already exists. Skipping.")
    else:
        sample_all_steps = xr.concat(
            array_list,
            dim="valid_time",
            data_vars="minimal",
            coords="different",
            compat="equals",
            combine_attrs="drop",
        ).sortby("valid_time")
        _logger.info(f"Saving to {out_fname}.")
        sample_all_steps = sample_all_steps.assign_coords(forecast_ref_time=frt)
        stream = str(sample_all_steps.coords["stream"].values)

        if "sample" in sample_all_steps.coords:
            sample_all_steps = sample_all_steps.drop_vars("sample")

        sample_all_steps = cf_parser_gaussian_aware(config, sample_all_steps)
        # Add Gaussian grid metadata if detected
        if "ncells" in sample_all_steps.dims:
            sample_all_steps = add_gaussian_grid_metadata(sample_all_steps)
            _logger.info("Detected and preserved Gaussian grid structure")
        # add forecast_period attributes
        n_hours = fstep_hours.astype("int64")
        sample_all_steps["forecast_period"] = sample_all_steps["forecast_step"] * n_hours
        sample_all_steps["forecast_period"].attrs = {
            "standard_name": "forecast_period",
            "long_name": "time since forecast_reference_time",
            "units": "hours",
        }
        sample_all_steps = add_conventions(stream, run_id, sample_all_steps)
        sample_all_steps.to_netcdf(out_fname, mode="w", compute=False)
