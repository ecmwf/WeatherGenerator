#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "weathergen-evaluate",
#   "weathergen-common",
#   "weathergen"
# ]
# [tool.uv.sources]
# weathergen-evaluate = { path = "../../../../../packages/evaluate" }
# weathergen-common = { path = "../../../../../packages/common" }
# weathergen = { path = "../../../../../" }
# ///
## Example USAGE: uv run export --run-id grwnhykd --stream ERA5 --output-dir /p/home/jusers/owens1/jureca/WeatherGen/test_output1 --format netcdf --type prediction target --fsteps 1 --samples 1
import argparse
import logging
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import xarray as xr
from omegaconf import OmegaConf
from tqdm import tqdm

from weathergen.common.config import _REPO_ROOT, get_model_results
from weathergen.common.io import ZarrIO

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

if not _logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    _logger.addHandler(handler)

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
        reshaped_dataset = reshaped_dataset.set_index(ipoint2=("ipoint", "valid_time")).unstack("ipoint2")
        #rename ipoint to ncells
        reshaped_dataset = reshaped_dataset.rename_dims({"ipoint": "ncells"})
        reshaped_dataset = reshaped_dataset.rename_vars({"ipoint": "ncells"})

    return reshaped_dataset


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
    ds.attrs["history"] = "Created using the export_inference.py script on " + np.datetime_as_string(
        np.datetime64("now"), unit="s"
    )
    ds.attrs["Conventions"] = "CF-1.12"
    return ds


def output_filename(
    prefix: str,
    run_id: str,
    output_dir: str,
    output_format: str,
    forecast_ref_time: np.datetime64,
) -> Path:
    """
    Generate output filename based on prefix (should refer to type e.g. pred/targ), run_id, sample index, output directory, format and forecast_ref_time.
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


def get_data(
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
        run_id : Run ID to identify the Zarr store.
        samples : Sample to process
        stream : Stream name to retrieve data for (e.g., 'ERA5').
        type : Type of data to retrieve ('target' or 'prediction').
        fsteps : List of forecast steps to retrieve. If None, retrieves all available forecast steps.
        channels :List of channels to retrieve. If None, retrieves all available channels.
        n_processes : Number of parallel processes to use for data retrieval.
        ecpoch : Epoch number to identify the Zarr store.
        rank : Rank number to identify the Zarr store.
        output_dir : Directory to save the NetCDF files.
        output_format : Output file format (currently only 'netcdf' supported).
        config : Loaded config for cf_parser function.
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
                    # reshape result - use adaptive function to handle both regular and Gaussian grids
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
        Dictionary where keys is sample index and values is a list of xarray DataArrays for all the forecast steps
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
    print(frt)
    out_fname = output_filename(type_str, run_id, output_dir, output_format, frt)
    # check if file already exists
    if out_fname.exists():
        _logger.info(f"File {out_fname} already exists. Skipping.")
    else:
        # remove multiple valid_times, uncomment later
        # def _clean_da(da: xr.DataArray) -> xr.DataArray:
        #     if "valid_time" in da.coords and da.valid_time.size > 1:
        #         unique_valid_time = np.unique(da.valid_time.values)
        #         da = da.drop_vars("valid_time")
        #         da = da.assign_coords(valid_time=unique_valid_time)
        #         # make valid_time a dimension for each variable
        #         #da = da.ass("ncells", "pressure_level", "valid_time")
        #         da = da.assign_coords()
        #     return da
        # array_list = [_clean_da(da) for da in array_list]
        # print(array_list[0])
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
        sample_all_steps["forecast_period"] = sample_all_steps["forecast_period"] * n_hours
        sample_all_steps["forecast_period"].attrs = {
            "standard_name": "forecast_period",
            "long_name": "time since forecast_reference_time",
            "units": "hours",
        }
        sample_all_steps = add_conventions(stream, run_id, sample_all_steps)
        sample_all_steps.to_netcdf(out_fname, mode="w", compute=False)


def parse_args(args: list) -> argparse.Namespace:
    """
    Parse command line arguments.

    Parameters
    ----------
        args : List of command line arguments.
    Returns
    -------
        Parsed command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id",
        type=str,
        help=" Zarr folder which contains target and inference results",
        required=True,
    )

    parser.add_argument(
        "--type",
        type=str,
        choices=["prediction", "target"],
        nargs="+",
        help="List of type of data to convert (e.g. prediction target)",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory to save the NetCDF files",
        required=True,
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["netcdf", "grib"],
        help="Output file format (currently only netcdf supported)",
        required=True,
    )

    parser.add_argument(
        "--stream",
        type=str,
        choices=["ERA5"],
        help="Stream name to retrieve data for",
        required=True,
    )

    parser.add_argument(
        "--fsteps",
        type=int,
        nargs="+",
        default=None,
        help="List of forecast steps to retrieve (e.g. 1 2 3). If not provided, retrieves all available forecast steps.",
    )

    parser.add_argument(
        "--samples",
        type=int,
        nargs="+",
        default=None,
        help="List of samples to process (e.g. 0 1 2). If not provided, processes all samples.",
    )

    parser.add_argument(
        "--channels",
        type=str,
        nargs="+",
        default=None,
        help="List of channels to retrieve (e.g., 'q_500 t_2m'). If not provided, retrieves all available channels.",
    )

    parser.add_argument(
        "--n-processes",
        type=int,
        default=8,
        help="Number of parallel processes to use for data retrieval",
    )

    parser.add_argument(
        "--fstep-hours",
        type=int,
        default=6,
        help="Time difference between forecast steps in hours (e.g., 6)",
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=0,
        help="Epoch number to identify the Zarr store",
    )

    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Rank number to identify the Zarr store",
    )

    args, unknown_args = parser.parse_known_args(args)
    if unknown_args:
        _logger.warning(f"Unknown arguments: {unknown_args}")
    return args


def export() -> None:
    """
    Main function to export data from Zarr store to NetCDF files.
    """
    # By default, arguments from the command line are read.
    export_from_args(sys.argv[1:])


def export_from_args(args: list) -> None:
    # Get run_id zarr data as lists of xarray DataArrays
    """
    Export data from Zarr store to NetCDF files based on command line arguments.
    Parameters
    ----------
        args : List of command line arguments.
    """
    args = parse_args(sys.argv[1:])
    run_id = args.run_id
    data_type = args.type
    output_dir = args.output_dir
    output_format = args.format
    samples = args.samples
    stream = args.stream
    fsteps = args.fsteps
    fstep_hours = np.timedelta64(args.fstep_hours, "h")
    channels = args.channels
    n_processes = args.n_processes
    epoch = args.epoch
    rank = args.rank

    # Ensure output directory exists
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    config_file = Path(_REPO_ROOT, "config/evaluate/config_zarr2cf.yaml")
    config = OmegaConf.load(config_file)
    # check config loaded correctly
    assert len(config["variables"].keys()) > 0, "Config file not loaded correctly"

    for dtype in data_type:
        _logger.info(f"Starting processing {dtype} for run ID {run_id}.")
        get_data(
            run_id,
            samples,
            stream,
            dtype,
            fsteps,
            channels,
            fstep_hours,
            n_processes,
            epoch,
            rank,
            output_dir,
            output_format,
            config,
        )
        _logger.info(f"Finished processing {dtype} for run ID {run_id}.")


if __name__ == "__main__":
    export()
