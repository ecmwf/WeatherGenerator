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
## Example USAGE: uv run export --run-id grwnhykd --stream ERA5 \
## --output-dir /p/home/jusers/owens1/jureca/WeatherGen/test_output1 \
## --format netcdf --type prediction target --fsteps 1 --samples 1
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from weathergen.common.config import _REPO_ROOT
from weathergen.evaluate.export.export_utils import get_data

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

if not _logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    _logger.addHandler(handler)


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
        help="List of forecast steps to retrieve (e.g. 1 2 3). "
        "If not provided, retrieves all available forecast steps.",
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
        help="List of channels to retrieve (e.g., 'q_500 t_2m'). "
        "If not provided, retrieves all available channels.",
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
