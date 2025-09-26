## EXAMPLE USAGE:
# uv run ./packages/evaluate/src/weathergen/evaluate/zarr_nc.py --run-id grwnhykd --stream ERA5 --output-dir /p/home/jusers/owens1/jureca/WeatherGen/test_output1 --format netcdf --type prediction target
import logging
import re
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import xarray as xr
from omegaconf import OmegaConf
from tqdm import tqdm

from weathergen.common.io import ZarrIO
from weathergen.utils.config import _load_private_conf

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

if not _logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    _logger.addHandler(handler)

## all functions
def find_pl(all_variables):
    """
    Find all the pressure levels for each variable and return a dictionary
    mapping variable names to their corresponding pressure levels.
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
    # rename sample
    sampled_data = [ds.assign_coords(sample=i) for i, ds in enumerate(sampled_data)]
    return sampled_data

def remove_ipoint(sample_data):
    """
    Remove ipoint dimension by setting it as index and unstacking.
    """
    sample_data = sample_data.set_index(ipoint=("valid_time", "lat", "lon")).unstack(
        "ipoint"
    )
    return sample_data

def add_conventions(stream, run_id, ds):
    """
    Add CF conventions to the dataset attributes.
    """
    ds.attrs["title"] = f"WeatherGenerator Output for {run_id} using stream {stream}"
    ds.attrs["institution"] = "WeatherGenerator Project"
    ds.attrs["source"] = "WeatherGenerator v0.0"
    ds.attrs["history"] = "Created using the zarr_nc.py script on " + np.datetime_as_string(np.datetime64("now"), unit="s")
    ds.attrs["Conventions"] = "CF-1.12"
    return ds

def cf_parser(config, ds) -> xr.Dataset:
    """
    Parse the dataset according to the CF conventions specified in the config.
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
        # TODO: better way to handle the ordering here
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
            # ds_attributes['variable'] = attributes
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

def output_filename(type, run_id, output_dir, output_format, forecast_ref_time):
    """
    Generate output filename based on type, run_id, sample index, output directory, format and forecast_ref_time.
    """
    if output_format not in ["netcdf"]:
        raise ValueError(f"Unsupported output format: {output_format}")
    file_extension = "nc"
    frt = np.datetime_as_string(forecast_ref_time, unit="h")
    # out_fname = Path(output_dir) / f'{type}_{run_id}_{np.datetime_as_string(forecast_ref_time, unit="h")}.{file_extension}'
    # documentation wants <type>_<forecast_reference_time>_<forecast_period>_<collection>.<extension> 
    #here multiple forecast steps are in one forecast so have amended to just use forecast reference time
    out_fname = (
        Path(output_dir) / f"{type}_{frt}_{run_id}.{file_extension}"
    )
    return out_fname

def zarr_store(run_id):
    """
    Get the path to the Zarr store for a given run ID.
    """
    run_results = (
        Path(_load_private_conf(None)["path_shared_working_dir"]) / f"results/{run_id}"
    )
    zarr_path = run_results / "validation_epoch00000_rank0000.zarr"
    # TODO: this might need to be more flexible for epochs and ranks
    if not zarr_path.exists() or not zarr_path.is_dir():
        raise FileNotFoundError(
            f"Zarr file {zarr_path} does not exist or is not a directory."
        )
    return zarr_path
    
def get_data(run_id: str, stream: str, type:str, fsteps=None, channels=None):
    """
    Retrieve data from Zarr store for a given run ID and stream.
    type: 'target' or 'prediction'
    """
    # check type is valid
    if type not in ['target', 'prediction']:
        raise ValueError(f"Invalid type: {type}. Must be 'target' or 'prediction'.")
    fname_zarr = zarr_store(run_id)
    with ZarrIO(fname_zarr) as zio:
        zio_forecast_steps = zio.forecast_steps
        samples = zio.samples
        dummy_out = zio.get_data(0, stream, zio_forecast_steps[0])
        all_channels = dummy_out.target.channels
        channels = all_channels if channels is None else channels

        fsteps = zio_forecast_steps if fsteps is None else fsteps
        fsteps = sorted([int(fstep) for fstep in fsteps])
        samples = sorted([int(sample) for sample in samples])
    da_list = []
    for fstep in range(1,  len(fsteps)):
        da_fs = []
        for sample in tqdm(
            samples,
            desc=f"Processing {run_id} - stream: {stream} - forecast step: {fstep}",
        ):
            with ZarrIO(fname_zarr) as zio:
                out = zio.get_data(sample, stream, fstep)
                if type == 'target':
                    data = out.target.as_xarray()
                elif type == 'prediction':
                    data = out.prediction.as_xarray()
            da_fs.append(data.squeeze())
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


def save_samples_to_netcdf(type_str, dict_sample_all_steps, FSTEP_HOURS, run_id, output_dir, output_format, config):
    """
    Uses dictionary of pred/target xarray DataArrays to save each sample to a NetCDF file.
    """
    for sample_idx, array_list in dict_sample_all_steps.items():
        frt = array_list[0].coords['valid_time'].values[0] - FSTEP_HOURS
        sample_all_steps = xr.concat(array_list, dim="forecast_step")
        out_fname = output_filename(
            type_str, run_id, output_dir, output_format, frt
        )
        _logger.info(f"Saving sample {sample_idx} to {out_fname}...")
        sample_all_steps = sample_all_steps.assign_coords(
            forecast_ref_time=frt
        )
        stream = str(sample_all_steps.coords['stream'].values)
        sample_all_steps = sample_all_steps.drop_vars("sample")
        sample_all_steps = cf_parser(config, sample_all_steps)
        sample_all_steps = add_conventions(stream, run_id, sample_all_steps)
        sample_all_steps.to_netcdf(out_fname, mode="w")

def parse_args(args):
    """
    Parse command line arguments.
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

    # run_id = 'grwnhykd'
    # stream = 'ERA5'
    # output_dir = './test_output'
    # output_format = 'netcdf'

    # Ensure output directory exists
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    # TODO: get path from platform-env

    config_file = "../WeatherGenerator-private/evaluate/config_zarr2cf.yaml"
    config = OmegaConf.load(config_file)
    # check config loaded correctly
    assert config["variables"]["q"] is not None

    FSTEP_HOURS = np.timedelta64(6, "h")

    # TODO: add checks to see if netcdf has alreaby been made
    # TODO: overwrite clobber = True
    # pverwriting needs to be fixed
    # PermissionError: [Errno 13] Permission denied: '/p/home/jusers/owens1/jureca/WeatherGen/test_output1/pred_grwnhykd_sample0.nc'
    # ds.close() needed to avoid this?
    # for now either wipe folder between runs or use new folder/don't insepct inbetween

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
        valid_time = [
            [np.unique(da.valid_time).squeeze() for da in da_step] for da_step in da_list
        ]
        forecast_ref_times = [vt - FSTEP_HOURS for vt in valid_time[0]]
        save_samples_to_netcdf(str(type)[:4], dict_sample_all_steps, FSTEP_HOURS, run_id, output_dir, output_format, config)