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
## Example USAGE: uv run export --run-id o8b60tgh --stream ERA5
# --output-dir ../o8b60tgh --format netcdf
# --regrid-degree 0.25 --regrid-type regular_ll
import argparse
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf

from weathergen.common.config import _REPO_ROOT
from weathergen.evaluate.export.export_core import export_model_outputs

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
    args :
        List of command line arguments.

    Returns
    -------
        Parsed command line arguments.
    """
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
        default=["prediction"],
        help="List of type of data to convert (e.g. prediction target)",
        required=False,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory to save the NetCDF files",
        required=True,
    )

    parser.add_argument(
        "--format",
        dest="output_format",
        type=str,
        choices=["netcdf", "grib", "quaver"],
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

    parser.add_argument(
        "--template",
        type=str,
        help="Path to GRIB template file",
        required=False,
    )

    parser.add_argument(
        "--expver",
        type=str,
        help="Expver to include in the output filename (i.e. 'iuoo')",
        required=False,
    )

    parser.add_argument(
        "--regrid-degree",
        type=float,
        default=None,
        help="""If specified, regrid the data to a regular lat/lon grid with the given degree,
        (e.g., 0.25 for 0.25x0.25 degree grid) or O/N Gaussian grid (e.g., 63 for N63 grid).""",
    )

    parser.add_argument(
        "--regrid-type",
        type=str,
        choices=["regular_ll", "O", "N"],
        default=None,
        help="Type of grid to regrid to (only used if --regrid-degree is specified)",
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

    # Load configuration
    config_file = Path(_REPO_ROOT, "config/evaluate/config_zarr2cf.yaml")
    config = OmegaConf.load(config_file)
    # check config loaded correctly
    assert len(config["variables"].keys()) > 0, "Config file not loaded correctly"

    kwargs = vars(args).copy()

    _logger.info(kwargs)

    # Ensure output directory exists
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for dtype in args.type:
        _logger.info(
            f"Starting processing {dtype} for run ID {args.run_id}. "
            f"Detected {args.samples} samples and {args.fsteps} forecast steps."
        )

        export_model_outputs(dtype, config, **kwargs)

        _logger.info(f"Finished processing {dtype} for run ID {args.run_id}.")


if __name__ == "__main__":
    export()
