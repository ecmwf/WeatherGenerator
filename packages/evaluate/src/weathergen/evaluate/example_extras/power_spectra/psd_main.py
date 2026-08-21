#!/usr/bin/env -S uv run --script
# ruff: noqa: E501
# /// script
# dependencies = [
#   "cf-units",
#   "scitools-iris>=3.11",
#   "weathergen-common",
#   "omegaconf"
# ]
#
# [tool.uv.sources]
# weathergen-common = { path = "../../../../../../common" }
# ///

"""
Plots the power spectrum of the analysis increments
Adapted from Martin Willet's code for power spectra
for use with the WeatherGenerator model:

.packages/evaluate/src/weathergen/evaluate/example_extras/power_spectra/psd_main.py
--run-id gn3gotvh --export-dir /p/home/jusers/owens1/juwels/WeatherGen/gn3gotvh

OR

./packages/evaluate/src/weathergen/evaluate/example_extras/power_spectra/psd_main.py \
--config ./packages/evaluate/src/weathergen/evaluate/example_extras/power_spectra/psd_config.yml

Prerequisties:

Please export the inference into a regular lat lon gridded netcdf first using the export package:
e.g.
uv run export --run-id <INFERENCE_ID> --stream ERA5 \
--output-dir ../output_nc --format netcdf --regrid-degree 1 \
--regrid-type regular_ll

Add the following line to the bashrc:
export LD_LIBRARY_PATH=/capstor/store/cscs/userlab/ch17/assets1/shared_libraries/udunits-2.2.28/lib:$LD_LIBRARY_PATH
"""

import argparse
import glob
import logging
import os
import sys
from pathlib import Path

import psd_plots as psd_plots
from omegaconf import DictConfig, OmegaConf

# Local application / package
from weathergen.common.config import _REPO_ROOT
from weathergen.common.logger import init_loggers

_logger = logging.getLogger(__name__)


def extract_filepaths(netcdf_paths: list) -> list:
    """
    Extracts filepaths from a list of netcdf paths.
    If a directory is given, all files in the directory are returned.
    Parameters
    ----------
    netcdf_paths:
        List of netcdf paths
    Returns
    -------
    list:
        List of filepaths
    """
    if len(netcdf_paths) > 1:
        # list of files
        return netcdf_paths
    else:
        netcdf_path = netcdf_paths[0]
        if os.path.isfile(netcdf_path):
            return netcdf_paths
        elif os.path.isdir(netcdf_path):
            glob_path = netcdf_path + "/*"
        else:
            glob_path = netcdf_path
        return glob.glob(glob_path)


def psd_from_config(cfg: dict) -> None:
    """
    Main function that controls power spectra density plotting.
    Parameters
    ----------
    cfg:
        Configuration input stored as dictionary
    """
    diags = cfg.variables
    regions = cfg.regions
    plevels = cfg.pressure_levels
    comparison_dict = {}
    for comp in cfg.comparisons:
        # extract file paths
        comparison_dict[comp] = extract_filepaths(cfg.comparisons[comp]["netcdf_paths"])
    outdir = cfg.output_dir
    os.makedirs(outdir, exist_ok=True)
    fname = cfg.prefix
    fc_times = cfg.forecast_steps

    psd_plots.plot_psds(
        comparison_dict,
        regions,
        diags,
        fname=fname,
        outdir=outdir,
        usencname=True,
        plevels=plevels,
        fc_times=fc_times,
    )


def parse_args(args: list) -> None:
    """
    Parse command line arguments.
    Parameters
    ----------
        args : List of command line arguments.
    """
    parser = argparse.ArgumentParser(description="Plot power spectral densities from NetCDF files.")

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the configuration YAML file.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=_REPO_ROOT / "plots" / "power_spectra",
        help="Directory to save the output plots.",
    )

    parser.add_argument(
        "--run-id",
        type=str,
        help="Run ID to construct configuration if --config is not provided.",
    )

    parser.add_argument(
        "--variables",
        type=str,
        nargs="+",
        help="List of variables to plot (e.g., 'u', 't2m'). If None, uses all",
        choices=["q", "t", "u", "v", "z", "t2m", "msl", "u10", "v10", "d2m", "skt", "sp"],
        default=["z", "u10", "v10"],
    )

    parser.add_argument(
        "--regions",
        type=str,
        nargs="+",
        help="List of regions to plot (e.g., 'ShortGlobe', 'N-Mid-Lats'). If None, uses all",
        choices=["FullGlobe", "ShortGlobe", "N-Mid-Lats", "S-Mid-Lats", "Tropics"],
        default=["ShortGlobe"],
    )

    parser.add_argument(
        "--pressure-levels",
        type=int,
        nargs="+",
        help="List of pressure levels to plot (e.g., 250, 500). \
               If not provided uses all",
        default=[100, 850],
    )

    parser.add_argument(
        "--forecast-steps",
        type=int,
        nargs="+",
        help="List of forecast steps to plot (e.g., 6, 12). \
               If not provided averages over all forecast steps",
        default=None,
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix for output files (default: empty).",
    )

    parser.add_argument(
        "--export-dir",
        type=str,
        help="Directory where exported NetCDF files were saved.",
        default=None,
    )

    args, unknown_args = parser.parse_known_args(args)
    if unknown_args:
        _logger.warning(f"Unknown arguments: {unknown_args}")
    return args


def construct_config_from_run_id(run_id: str, args: argparse.Namespace) -> DictConfig:
    """
    Construct configuration from run ID and command line arguments.
    Parameters
    ----------
        run_id : Run ID to construct configuration for.
        args : Command line arguments.
    Returns
    -------
        DictConfig: Constructed configuration.
    """
    run_id_config = {
        "variables": args.variables,
        "regions": args.regions,
        "pressure_levels": args.pressure_levels,
        "forecast_steps": args.forecast_steps,
        "prefix": args.prefix,
        "output_dir": Path(args.output_dir),
        "comparisons": {
            "target": {"netcdf_paths": [f"{args.export_dir}/targ*.nc"]},
            run_id: {"netcdf_paths": [f"{args.export_dir}/pred*.nc"]},
        },
    }
    run_id_config = DictConfig(run_id_config)
    return run_id_config


def psd_from_args(args: list) -> None:
    # Get run_id zarr data as lists of xarray DataArrays
    """
    Export data from Zarr store to NetCDF files based on command line arguments.
    Parameters
    ----------
        args : List of command line arguments.
    """
    init_loggers()

    args = parse_args(sys.argv[1:])

    # Load configuration
    if args.config:
        config_file = Path(args.config)
        config = OmegaConf.load(config_file)
        # check config loaded correctly
        assert isinstance(config, DictConfig), "Config file not loaded correctly"
        # use PosixPath for output_dir
        config.output_dir = Path(config.output_dir)

    # Use run id to construct config if not provided
    elif args.run_id:
        if args.export_dir is None:
            # TODO: automatically run export into results directory and use that path here
            raise ValueError("When using --run-id, --export-dir must also be provided.")
        config = construct_config_from_run_id(args.run_id, args)

    else:
        raise ValueError("Either --config or --run-id must be provided.")

    _logger.info(f"starting power spectral density plotting with config: {config}")

    psd_from_config(config)


def psd() -> None:
    """
    Main function to plot power spectral densities.
    """
    # By default, arguments from the command line are read.
    psd_from_args(sys.argv[1:])


if __name__ == "__main__":
    psd()
