## EXAMPLE USAGE:
# uv run ./packages/evaluate/src/weathergen/evaluate/zarr_nc.py --run-id grwnhykd --stream ERA5 --output-dir /p/home/jusers/owens1/jureca/WeatherGen/test_output1 --format netcdf --type prediction target
import logging
import re
import sys
from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import xarray as xr
from omegaconf import OmegaConf
from tqdm import tqdm

from weathergen.common.config import _REPO_ROOT, _load_private_conf
from weathergen.common.io import ZarrIO

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

if not _logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    _logger.addHandler(handler)


def find_pl(all_variables):
    """
    Find all the pressure levels for each variable using regex and returns a dictionary
    mapping variable names to their corresponding pressure levels.
    Parameters
    ----------
    all_variables : list of str
        List of variable names with pressure levels (e.g.,'q_500','t_2m').
    Returns
    -------
    tuple
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


def reshape_dataset(input_data_array):
    """
    Reshape the input xarray DataArray to have dimensions (ipoint, pressure_level)
    for variables with multiple pressure levels, and (ipoint,) for surface variables.
    Removes ipoint to valid_time, lat, lon after splitting for each sample
    Parameters
    ----------
    input_data_array : xarray.DataArray
        Input xarray DataArray with dimensions (ipoint, channel).
    Returns
    -------
    list of xarray.Dataset
        List of xarray Datasets, one for each sample, with reshaped dimensions.
    """
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
                input_data_array.sel(channel=old_vars[0]).values, dims=["ipoint"]
            )
    reshaped_dataset = xr.Dataset(data_vars)
    reshaped_dataset = reshaped_dataset.assign_coords(
        ipoint=input_data_array.coords["ipoint"],
        pressure_level=pl,
    )
    sampled_data = [
        reshaped_dataset.where(reshaped_dataset.sample == i, drop=True)
        for i in range(len(np.unique(reshaped_dataset.sample)))
    ]
    sampled_data = [remove_ipoint(sample) for sample in sampled_data]
    sampled_data = [ds.assign_coords(sample=i) for i, ds in enumerate(sampled_data)]
    return sampled_data


def remove_ipoint(sample_data):
    """
    Remove ipoint dimension by setting it as index and unstacking.
    Parameters
    ----------
    sample_data : xarray.Dataset
        Input xarray Dataset with ipoint dimension.
    Returns
    -------
    xarray.Dataset
        xarray Dataset without ipoint dimension (replaced by valid_time, lat and lon).
    """
    sample_data = sample_data.set_index(ipoint=("valid_time", "lat", "lon")).unstack(
        "ipoint"
    )
    return sample_data


def add_conventions(stream, run_id, ds):
    """
    Add CF conventions to the dataset attributes.
    Parameters
    ----------
    stream : str
        Stream name to include in the title attribute.
    run_id : str
        Run ID to include in the title attribute.
    ds : xarray.Dataset
        Input xarray Dataset to add conventions to.
    Returns
    -------
    xarray.Dataset
        xarray Dataset with CF conventions added to attributes.
    """
    ds.attrs["title"] = f"WeatherGenerator Output for {run_id} using stream {stream}"
    ds.attrs["institution"] = "WeatherGenerator Project"
    ds.attrs["source"] = "WeatherGenerator v0.0"
    ds.attrs["history"] = (
        "Created using the zarr_nc.py script on "
        + np.datetime_as_string(np.datetime64("now"), unit="s")
    )
    ds.attrs["Conventions"] = "CF-1.12"
    return ds


def cf_parser(config, ds) -> xr.Dataset:
    """
    Parse the dataset according to the CF conventions specified in the config.
    Parameters
    ----------
    config : OmegaConf
        Loaded config for cf_parser function.
    ds : xarray.Dataset
        Input xarray Dataset to be parsed.
    Returns
    -------
    xarray.Dataset
        Parsed xarray Dataset with CF conventions applied.
    """
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
            if "std_unit" in dim_dict and dim_dict["std_unit"] is not None:
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
        dims = [
            "forecast_period",
            "pressure",
            "valid_time",
            "latitude",
            "longitude",
        ]
        if mapping[var_name]["level_type"] == "sfc":
            dims.remove("pressure")
        coordinates = {}
        for coord, new_name in config["coordinates"][
            mapping[var_name]["level_type"]
        ].items():
            coordinates |= {
                new_name: (
                    ds.coords[coord].dims,
                    ds.coords[coord].values,
                    ds_attributes[new_name],
                )
            }
        try:
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
        except Exception as e:
            _logger.info("Problem with ", e)
    dataset = xr.merge(variables.values())
    dataset.attrs = ds.attrs
    return dataset


def output_filename(prefix, run_id, output_dir, output_format, forecast_ref_time):
    """
    Generate output filename based on prefix (should refer to type e.g. pred/targ), run_id, sample index, output directory, format and forecast_ref_time.
    Parameters
    ----------
    prefix : str
        Prefix for file name (e.g., 'pred' or 'targ').
    run_id : str
        Run ID to include in the filename.
    output_dir : str
        Directory to save the output file.
    output_format : str
        Output file format (currently only 'netcdf' supported).
    forecast_ref_time : np.datetime64
        Forecast reference time to include in the filename.
    Returns
    -------
    Path
        Full path to the output file.
    """
    if output_format not in ["netcdf"]:
        raise ValueError(f"Unsupported output format: {output_format}")
    file_extension = "nc"
    frt = np.datetime_as_string(forecast_ref_time, unit="h")
    out_fname = Path(output_dir) / f"{prefix}_{frt}_{run_id}.{file_extension}"
    return out_fname


def zarr_store(run_id):
    """
    Get the path to the Zarr store for a given run ID.
    Parameters
    ----------
    run_id : str
        Run ID to identify the Zarr store.
    Returns
    -------
    Path
        Path to the Zarr store.
    """
    run_results = (
        Path(_load_private_conf(None)["path_shared_working_dir"]) / f"results/{run_id}"
    )
    zarr_path = run_results / "validation_epoch00000_rank0000.zarr"
    if not zarr_path.exists() or not zarr_path.is_dir():
        raise FileNotFoundError(
            f"Zarr file {zarr_path} does not exist or is not a directory."
        )
    return zarr_path


def get_data_worker(args):
    """
    Worker function to retrieve data for a single sample and forecast step.
    Parameters
    ----------
    args : tuple
        Tuple containing (sample, fstep, run_id, stream, type).
    Returns
    -------
    xarray.DataArray
        xarray DataArray for the specified sample and forecast step.
    """
    sample, fstep, run_id, stream, type = args
    fname_zarr = zarr_store(run_id)
    try:
        with ZarrIO(fname_zarr) as zio:
            out = zio.get_data(sample, stream, fstep)
            if type == "target":
                data = out.target.as_xarray()
            elif type == "prediction":
                data = out.prediction.as_xarray()
        return data.squeeze()
    except Exception as e:
        _logger.error(
            f"[get_data_worker ERROR] sample={sample}, fstep={fstep} failed: {e}"
        )
        return None


def get_data(
    run_id: str, stream: str, type: str, fsteps=None, channels=None, n_processes=4
):
    """
    Retrieve data from Zarr store and return as a list of xarray DataArrays for each forecast step.
    Using multiprocessing to speed up data retrieval.

    Parameters
    ----------
    run_id : str
        Run ID to identify the Zarr store.
    stream : str
        Stream name to retrieve data for (e.g., 'ERA5').
    type : str
        Type of data to retrieve ('target' or 'prediction').
    fsteps : list of int, optional
        List of forecast steps to retrieve. If None, retrieves all available forecast steps.
    channels : list of str, optional
        List of channels to retrieve. If None, retrieves all available channels.
    Returns
    -------
    list of xarray.DataArray
        List of xarray DataArrays for each forecast step.

    """
    if type not in ["target", "prediction"]:
        raise ValueError(f"Invalid type: {type}. Must be 'target' or 'prediction'.")

    fname_zarr = zarr_store(run_id)
    with ZarrIO(fname_zarr) as zio:
        zio_forecast_steps = sorted([int(step) for step in zio.forecast_steps])
        samples = sorted([int(sample) for sample in zio.samples])
        dummy_out = zio.get_data(0, stream, zio_forecast_steps[0])
        all_channels = dummy_out.target.channels
        channels = all_channels if channels is None else channels

    fsteps = (
        zio_forecast_steps
        if fsteps is None
        else sorted([int(fstep) for fstep in fsteps])
    )

    da_list = []
    for fstep in fsteps:
        step_tasks = [(sample, fstep, run_id, stream, type) for sample in samples]
        da_fs = []

        with Pool(processes=n_processes) as pool:
            for result in tqdm(
                pool.imap_unordered(get_data_worker, step_tasks, chunksize=1),
                total=len(step_tasks),
                desc=f"Processing {run_id} - stream: {stream} - forecast step: {fstep}",
            ):
                if result is not None:
                    da_fs.append(result)

        if da_fs:
            da_fs = xr.concat(da_fs, dim="ipoint")
            if set(channels) != set(all_channels):
                available_channels = da_fs.channel.values
                existing_channels = [ch for ch in channels if ch in available_channels]
                if len(existing_channels) < len(channels):
                    _logger.info(
                        f"The following channels were not found: {list(set(channels) - set(existing_channels))}. Skipping them."
                    )
                da_fs = da_fs.sel(channel=existing_channels)
            da_list.append(da_fs)

    return da_list


def save_samples_to_netcdf(
    type_str,
    dict_sample_all_steps,
    FSTEP_HOURS,
    run_id,
    output_dir,
    output_format,
    config,
):
    """
    Uses dictionary of pred/target xarray DataArrays to save each sample to a NetCDF file.
    Parameters
    ----------
    type_str : str
        Type of data ('pred' or 'targ') to include in the filename.
    dict_sample_all_steps : dict
        Dictionary where keys are sample indices and values are lists of xarray DataArrays for all the forecast steps
    FSTEP_HOURS : np.timedelta64
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
    for sample_idx, array_list in dict_sample_all_steps.items():
        frt = array_list[0].coords["valid_time"].values[0] - FSTEP_HOURS
        sample_all_steps = xr.concat(array_list, dim="forecast_step")
        out_fname = output_filename(type_str, run_id, output_dir, output_format, frt)
        _logger.info(f"Saving sample {sample_idx} to {out_fname}...")
        sample_all_steps = sample_all_steps.assign_coords(forecast_ref_time=frt)
        stream = str(sample_all_steps.coords["stream"].values)
        sample_all_steps = sample_all_steps.drop_vars("sample")
        sample_all_steps = cf_parser(config, sample_all_steps)
        sample_all_steps = add_conventions(stream, run_id, sample_all_steps)
        sample_all_steps.to_netcdf(out_fname, mode="w")


def parse_args(args):
    """
    Parse command line arguments.

    Parameters
    ----------
    args : list of str
        List of command line arguments.
    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--run-id",
        type=str,
        help=" Zarr folder which contains target and inference results",
    )

    parser.add_argument(
        "--type",
        type=str,
        choices=["prediction", "target"],
        nargs="+",
        help="List of type of data to convert (e.g. prediction target)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory to save the NetCDF files",
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["netcdf", "grib"],
        help="Output file format (currently only netcdf supported)",
    )

    parser.add_argument(
        "--stream",
        type=str,
        choices=["ERA5"],
        help="Stream name to retrieve data for",
    )
    args, unknown_args = parser.parse_known_args(args)
    if unknown_args:
        _logger.warning(f"Unknown arguments: {unknown_args}")
    return args


if __name__ == "__main__":
    # Get run_id zarr data as lists of xarray DataArrays
    args = parse_args(sys.argv[1:])
    run_id = args.run_id
    data_type = args.type
    output_dir = args.output_dir
    output_format = args.format
    stream = args.stream

    # Ensure output directory exists
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    config_file = Path(_REPO_ROOT, "config/evaluate/config_zarr2cf.yaml")
    config = OmegaConf.load(config_file)
    # check config loaded correctly
    assert config["variables"]["q"] is not None

    FSTEP_HOURS = np.timedelta64(6, "h")

    for type in data_type:
        _logger.info(f"Starting processing {type} for run ID {run_id}.")
        da_list = get_data(run_id, stream, type)
        n_samples = len(np.unique(da_list[0].sample))
        dict_sample_all_steps = {}
        for i, da_type in enumerate(da_list):
            fs_i_all_sample = reshape_dataset(da_type)
            for j in range(n_samples):
                _logger.info(f"Processing sample {j}, forecast step {i + 1}")
                dict_sample_all_steps.setdefault(j, []).append(fs_i_all_sample[j])
        #check dict_sample_all_steps is not empty
        assert dict_sample_all_steps, "No data to save, dict_sample_all_steps is empty."
        try:
            _logger.info(f"Saving {type} data to {output_format} format in {output_dir}.")
            save_samples_to_netcdf(
                str(type)[:4],
                dict_sample_all_steps,
                FSTEP_HOURS,
                run_id,
                output_dir,
                output_format,
                config,
            )#
        except Exception as e:
            _logger.error(f"Error saving {type} data: {e}")
        _logger.info(f"Finished processing {type} for run ID {run_id}.")
