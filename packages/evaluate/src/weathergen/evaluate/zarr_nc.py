## EXAMPLE USAGE:
# uv run ./packages/evaluate/src/weathergen/evaluate/zarr_nc.py --run-id grwnhykd --stream ERA5 --output-dir /p/home/jusers/owens1/jureca/WeatherGen/test_output1 --format netcdf --type prediction target
import logging
import os
import re
import subprocess
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
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    _logger.addHandler(handler)

def find_pl(all_variables):
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
    var_dict, pl = find_pl(input_data_array.channel.values)
    data_vars = {}
    for new_var, old_vars in var_dict.items():
        if len(old_vars) > 1:
            data_vars[new_var] = xr.DataArray(
                input_data_array.sel(channel=old_vars).values,
                dims=['ipoint', 'pressure_level']
            )
        else:
            data_vars[new_var] = xr.DataArray(
                input_data_array.sel(channel=old_vars[0]).values,
                dims=['ipoint']
            )
    reshaped_dataset = xr.Dataset(data_vars)
    reshaped_dataset = reshaped_dataset.assign_coords(
        ipoint=input_data_array.coords['ipoint'],
        pressure_level=pl,
    )
    sampled_data = [reshaped_dataset.where(reshaped_dataset.sample == i, drop=True) for i in range(len(np.unique(reshaped_dataset.sample)))]
    #print(sampled_data[1])
    #print(remove_ipoint(sampled_data[1]))
    sampled_data = [remove_ipoint(sample) for sample in sampled_data]
    # rename sample
    sampled_data = [ds.assign_coords(sample=i) for i, ds in enumerate(sampled_data)]
    return sampled_data

def remove_ipoint(sample_data):
    sample_data = sample_data.set_index(ipoint=("valid_time", "lat", "lon")).unstack("ipoint")
    return sample_data

def add_conventions(run_id, ds):
    ds.attrs["title"] = f"WeatherGenerator Output for {run_id}"
    ds.attrs["institution"] = "WeatherGenerator Project"
    ds.attrs["source"] = "WeatherGenerator v0.0"
    ds.attrs["history"] = "none"
    ds.attrs["Conventions"] = "CF-1.12"
    return ds

def cf_parser(config, ds) -> xr.Dataset:
    # Start a new xarray dataset from scratch, it's easier than deleting / renaming (I tried!).
    variables = {}
    mapping = config["variables"]

    ds_attributes = {}
    for dim_name, dim_dict in config["dimensions"].items():
        # clear dimensions if key and dim_dict['wg'] are the same
        if dim_name == dim_dict['wg']:
            dim_attributes = dict(
                standard_name = dim_dict.get('std',None),
            )
            if 'std_unit' in dim_dict and dim_dict['std_unit'] is not None:
                dim_attributes['units'] = dim_dict['std_unit']
            ds_attributes[dim_dict['wg']] = dim_attributes
            continue
        if dim_name in ds.dims:
            ds = ds.rename_dims({dim_name: dim_dict['wg']})
        dim_attributes = dict(
                    standard_name = dim_dict.get('std',None),
                    units = dim_dict.get('std_unit',None)
                )
        ds_attributes[dim_dict['wg']] = dim_attributes
    for var_name in ds:
        # TODO: better way to handle the ordering here
        dims = ['forecast_step', 'pressure', 'valid_time', 'latitude', 'longitude',]
        if mapping[var_name]["level_type"] == "sfc":
            dims.remove('pressure')
        coordinates = {}
        for coord, new_name in config["coordinates"][mapping[var_name]["level_type"]].items():
            coordinates |= {new_name: (ds.coords[coord].dims, ds.coords[coord].values, ds_attributes[new_name])}
        try:
            variable = ds[var_name]
            attributes = dict(
                standard_name=mapping[var_name]["std"],
                units=mapping[var_name]["std_unit"]
            )
            # ds_attributes['variable'] = attributes
            variables[mapping[var_name]["var"]] = xr.DataArray(
                data=variable.values,
                dims=dims,                
                coords={**coordinates, 'valid_time': ds['valid_time'].values},
                attrs=attributes,
                name=mapping[var_name]["var"]
            )
        except Exception as e:
            _logger.info('Problem with ', e)
    dataset = xr.merge(variables.values())
    dataset.attrs = ds.attrs
    return dataset

def output_filename(type, run_id, sample_idx, output_dir, output_format):
    if output_format not in ['netcdf']:
        raise ValueError(f"Unsupported output format: {output_format}")
    file_extension = 'nc'
    #out_fname = Path(output_dir) / f'{type}_{run_id}_{np.datetime_as_string(forecast_ref_time, unit="h")}.{file_extension}'
    out_fname = Path(output_dir) / f'{type}_{run_id}_sample{sample_idx}.{file_extension}'
    return out_fname

def zarr_store(run_id):
    run_results = Path(_load_private_conf(None)["path_shared_working_dir"]) / f"results/{run_id}"
    zarr_path = run_results / 'validation_epoch00000_rank0000.zarr'
    if not zarr_path.exists() or not zarr_path.is_dir():
        raise FileNotFoundError(f"Zarr file {zarr_path} does not exist or is not a directory.")
    return zarr_path

def get_data1(run_id: str, stream: str, fsteps = None, channels = None):
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

        da_tars, da_preds = [], []

        for fstep in range(1,len(fsteps)):
            da_tars_fs, da_preds_fs = [], []

            for sample in tqdm(samples, desc=f"Processing {run_id} - stream: {stream} - forecast step: {fstep}"):
                out = zio.get_data(sample, stream, fstep)
                target, pred = out.target.as_xarray(), out.prediction.as_xarray()
                da_tars_fs.append(target.squeeze())
                da_preds_fs.append(pred.squeeze())

            da_tars_fs = xr.concat(da_tars_fs, dim="ipoint")
            da_preds_fs = xr.concat(da_preds_fs, dim="ipoint")

            if set(channels) != set(all_channels):
                available_channels = da_tars_fs.channel.values
                existing_channels = [ch for ch in channels if ch in available_channels]
                if len(existing_channels) < len(channels):
                    _logger.info(f"The following channels were not found: {list(set(channels) - set(existing_channels))}. Skipping them.")
                da_tars_fs = da_tars_fs.sel(channel=existing_channels)
                da_preds_fs = da_preds_fs.sel(channel=existing_channels)

            da_tars.append(da_tars_fs)
            da_preds.append(da_preds_fs)

        return da_tars, da_preds

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
        choices=['prediction', 'target'],
        nargs='+',
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
        choices=['netcdf', 'grib'],
        help="Output file format (currently only netcdf supported)",
    )

    parser.add_argument(
        "--stream",
        type=str,
        choices=['ERA5'],
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
    config_file = '../WeatherGenerator-private/evaluate/config_zarr2cf.yaml'
    config = OmegaConf.load(config_file)
    # check config loaded correctly
    assert config['variables']['q'] is not None

    da_tars, da_preds = get_data1(run_id, stream)

    n_samples = len(np.unique(da_tars[0].sample))
    fstep_hours = np.timedelta64(6, "h")

    valid_time = [[np.unique(da.valid_time).squeeze() for da in da_pred] for da_pred in da_preds]
    steps = [np.unique(da_pred[0].forecast_step.values) for da_pred in da_preds]
    forecast_ref_time = [vt - fstep_hours for vt in valid_time[0]]

    # Store predictions and targets per sample
    pred_sample_all_steps = {}
    targ_sample_all_steps = {}

    for i, da_pred in enumerate(da_preds):
        fs_i_all_sample_pred = reshape_dataset(da_pred)
        for j in range(n_samples):
            # TODO: check forecast step values and ordering
            _logger.info(f"Processing sample {j}, forecast step {i+1}")
            pred_sample_all_steps.setdefault(j, []).append(fs_i_all_sample_pred[j])

    for i, da_targ in enumerate(da_tars):
        fs_i_all_sample_targ = reshape_dataset(da_targ)
        for j in range(n_samples):
            _logger.info(f"Processing sample {j}, forecast step {i+1}")
            targ_sample_all_steps.setdefault(j, []).append(fs_i_all_sample_targ[j])

    # Save each sample's predictions
    for sample_idx, pred_list in pred_sample_all_steps.items():
        sample_all_steps = xr.concat(pred_list, dim='forecast_step')
        # TODO: put forecast reference time in the filename
        out_fname = output_filename('pred', run_id, sample_idx, output_dir, output_format)
        _logger.info(f"Saving sample {sample_idx} to {out_fname}...")
        sample_all_steps = sample_all_steps.assign_coords(forecast_ref_time=forecast_ref_time[sample_idx])
        sample_all_steps = sample_all_steps.drop_vars('sample')
        sample_all_steps = sample_all_steps.assign_coords(forecast_period=sample_all_steps.forecast_step * fstep_hours)
        sample_all_steps = cf_parser(config, sample_all_steps)
        sample_all_steps = add_conventions(run_id, sample_all_steps)
        sample_all_steps.to_netcdf(out_fname, mode = 'w')

    # Save each sample's targets
    # TODO: filter useing data_type, make functions
    for sample_idx, targ_list in targ_sample_all_steps.items():
        sample_all_steps = xr.concat(targ_list, dim='forecast_step')
        out_fname = output_filename('targ', run_id, sample_idx, output_dir, output_format)
        _logger.info(f"Saving sample {sample_idx} to {out_fname}...")
        sample_all_steps = sample_all_steps.assign_coords(forecast_ref_time=forecast_ref_time[sample_idx])
        sample_all_steps = sample_all_steps.drop_vars('sample')
        sample_all_steps = sample_all_steps.assign_coords(forecast_period=sample_all_steps.forecast_step * fstep_hours)
        sample_all_steps = cf_parser(config, sample_all_steps)
        sample_all_steps = add_conventions(run_id, sample_all_steps)
        sample_all_steps.to_netcdf(out_fname, mode = 'w')
        # pverwriting needs to be fixed
        # PermissionError: [Errno 13] Permission denied: '/p/home/jusers/owens1/jureca/WeatherGen/test_output1/pred_grwnhykd_sample0.nc'
        # ds.close() needed to avoid this? 
        # for now either wipe folder between runs or use new folder/don't insepct inbetween