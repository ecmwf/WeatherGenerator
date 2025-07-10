#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "weathergen-evaluate",
#   "weathergen-common",
# ]
# [tool.uv.sources]
# weathergen-evaluate = { path = "../../../../../packages/evaluate" }
# ///

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from weathergen.common.io import ZarrIO
from weathergen.evaluate.score import VerifiedData, get_score

_logger = logging.getLogger(__name__)

_REPO_ROOT = Path(
    __file__
).parent.parent.parent.parent.parent  # TODO use importlib for resources
_DEFAULT_RESULT_PATH = _REPO_ROOT / "results"


### Auxiliary functions
def peek_tar_channels(zio: ZarrIO, stream: str):
    """
    Peek the channels of a tar stream in a ZarrIO object.

    Parameters
    ----------
    zio : ZarrIO
        The ZarrIO object containing the tar stream.
    stream : str
        The name of the tar stream to peek.
    Returns
    -------
    channels : list
        A list of channel names in the tar stream.
    """
    if not isinstance(zio, ZarrIO):
        raise TypeError("zio must be an instance of ZarrIO")

    dummy_out = zio.get_data(0, stream, 0)
    channels = dummy_out.target.channels

    return channels


def calc_scores_per_stream(
    zio: ZarrIO, stream: str, metrics: list[str], channels: str | list[str] = None
):
    """
    Calculate the provided score metrics for a specific.
    Parameters
    ----------
    zio : ZarrIO
        The ZarrIO object containing the data.
    stream : str
        The name of the stream to process.
    metrics : list[str]
        A list of metric names to calculate.
    Returns
    -------
    metric_stream : xr.DataArray
        An xarray DataArray containing the computed metrics for the specified stream.
    """
    # Get stream-specific information
    forecast_steps = zio.forecast_steps
    nmetrics, nforecast_steps = len(metrics), len(forecast_steps)
    samples = [int(sample) for sample in zio.samples]
    channels_stream = peek_tar_channels(zio, stream)
    # filter channels if provided
    channels_stream = (
        [ch for ch in channels_stream if ch in to_list(channels)]
        if channels
        else channels_stream
    )

    # initialize the DataArray to store metrics
    metric_stream = xr.DataArray(
        np.full((nforecast_steps, len(channels_stream), nmetrics), np.nan),
        coords={
            "forecast_step": forecast_steps,
            "channel": channels_stream,
            "metric": metrics,
        },
    )

    print(f"Processing stream {stream}...")
    for fstep in forecast_steps:
        targets, preds = [], []
        _logger.info(f"Processing forecast_step {fstep} of stream {stream}...")
        for sample in sorted(samples)[0:10]:
            out = zio.get_data(sample, stream, fstep)
            targets.append(out.target.as_xarray().squeeze())
            preds.append(out.prediction.as_xarray().squeeze())

        # Concatenate targets and predictions along the 'ipoint' dimension and verify the data
        _logger.debug(
            f"Concatenating targets and predictions for stream {stream}, forecast_step {fstep}..."
        )
        targets_all, preds_all = (
            xr.concat(targets, dim="ipoint"),
            xr.concat(preds, dim="ipoint"),
        )
        _logger.debug(f"Verifying data for stream {stream}, forecast_step {fstep}...")
        score_data = VerifiedData(preds_all, targets_all)

        # Build up computation graphs for all metrics
        _logger.debug(
            f"Build computation graphs for metrics for stream {stream}, forecast_step {fstep}..."
        )
        combined_metrics = [
            get_score(score_data, metric, agg_dims="ipoint") for metric in metrics
        ]
        combined_metrics = xr.concat(combined_metrics, dim="metric")
        combined_metrics["metric"] = metrics

        # Store the computed metrics in the DataArray and does computation
        metric_stream.loc[{"forecast_step": fstep}] = combined_metrics.compute()
        _logger.info(f"Computed metrics for forecast_step {fstep} of stream {stream}.")

    return metric_stream


def to_list(obj: Any) -> list:
    """
    Convert given object to list if obj is not already a list. Sets are also transformed to a list.

    Parameters
    ----------
    obj : Any
        The object to transform into a list.
    Returns
    -------
    list
        A list containing the object, or the object itself if it was already a list.
    """
    if isinstance(obj, set | tuple):
        obj = list(obj)
    elif not isinstance(obj, list):
        obj = [obj]
    return obj


def metric_list_to_dict(metric_list: list[xr.DataArray], streams: list):
    """
    Convert a list of xarray DataArrays containing metrics into a nested dictionary structure.

    Parameters
    ----------
    metric_list : list[xr.DataArray]
        A list of xarray DataArrays, each containing metrics for a specific stream.
    streams : list[str]
        A list of stream names corresponding to the DataArrays in metric_list.
    Returns
    -------
    result : dict
        A nested dictionary where the first level keys are stream names, the second level keys are channel names (if applicable),
        and the third level keys are forecast steps, with metric names as the final keys.
    """
    result = {}

    assert len(metric_list) == len(streams), (
        "Inconsistent list of metrics and streams passed."
    )

    for istream, da_metric in enumerate(metric_list):
        # Get the stream name (e.g., 'ERA5', 'IMERG', etc.)
        stream = streams[istream]
        if stream not in result:
            result[stream] = {}

        # Check if 'channel' exists (not all arrays have it)
        has_channel = "channel" in da_metric.dims

        metrics = da_metric.coords["metric"].values
        forecast_steps = to_list(da_metric.coords["forecast_step"].values)

        if has_channel:
            channels = da_metric.coords["channel"].values
            for ch_idx, ch in enumerate(channels):
                ch = str(ch)
                if ch not in result[stream]:
                    result[stream][ch] = {}

                for step_idx, step in enumerate(forecast_steps):
                    step = str(step)
                    if step not in result[stream][ch]:
                        result[stream][ch][step] = {}

                    for metric_idx, metric in enumerate(metrics):
                        metric = str(metric)
                        value = float(
                            da_metric.isel(
                                {
                                    "metric": metric_idx,
                                    "forecast_step": step_idx,
                                    "channel": ch_idx,
                                }
                            ).values
                        )
                        result[stream][ch][step][metric] = value
        else:
            # If no channel dimension, data is per forecast_step and metric only
            for step_idx, step in enumerate(forecast_steps):
                step = str(step)
                if step not in result[stream]:
                    result[stream][step] = {}

                for metric_idx, metric in enumerate(metrics):
                    metric = str(metric)
                    value = float(da_metric.values[metric_idx, step_idx])
                    result[stream][step][metric] = value

    return result


def fast_evaluation(
    run_id: str,
    metrics: list[str],
    save_dir: Path,
    results_dir: Path = _DEFAULT_RESULT_PATH,
    streams: str | list[str] = None,
    channels: str | list[str] = None,
    epoch: int = 0,
    rank: int = 0,
):
    """
    Perform fast evaluation of a run using the specified metrics and save the results.

    Parameters
    ----------
    run_id : str
        The ID of the run to evaluate.
    metrics : list[str]
        A list of metric names to evaluate.
    save_dir : Path
        The directory where the JSON-file with the results will be saved.
    """

    # get path to zarr storage
    results_zarr = (
        results_dir / run_id / f"validation_epoch{epoch:05d}_rank{rank:04d}.zarr"
    )

    if not results_zarr.exists():
        raise FileNotFoundError(f"Results zarr file not found: {results_zarr}")

    all_metric_streams = []

    # Open the ZarrIO object to access the data
    _logger.info(f"Loading inference data from{results_zarr}")

    with ZarrIO(results_zarr) as zio:
        streams = streams or zio.streams

        for stream in streams:
            _logger.info(f"Processing stream {stream}...")

            metric_stream = calc_scores_per_stream(zio, stream, metrics, channels)
            all_metric_streams.append(metric_stream)

    _logger.info(
        f"Finished computing metric scores for all streams. Total streams processed: {len(all_metric_streams)}"
    )

    # Convert the list of metric streams to a nested dictionary structure
    _logger.debug("Converting metric streams to dictionary format...")
    metric_dict = metric_list_to_dict(all_metric_streams, streams)

    # Save the results to a JSON file
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"metrics_{run_id}_epoch{epoch:05d}_rank{rank:04d}.json"

    _logger.info(f"Saving results to {save_path}")
    with open(save_path, "w") as f:
        json.dump(metric_dict, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fast evaluation of weather generator runs."
    )

    parser.add_argument(
        "-id", "--run_id", type=str, help="The ID of the run to evaluate."
    )
    parser.add_argument(
        "-m",
        "--metrics",
        type=str,
        nargs="+",
        default=["rmse", "mae", "bias"],
        help="List of metrics to evaluate.",
    )
    parser.add_argument(
        "-sd",
        "--save_dir",
        type=Path,
        default=None,
        help="Directory to save the results.",
    )
    parser.add_argument(
        "-rd",
        "--results_dir",
        type=Path,
        default=_DEFAULT_RESULT_PATH,
        help="Directory containing the results zarr files.",
    )
    parser.add_argument(
        "-s",
        "--streams",
        type=str,
        nargs="*",
        default=None,
        help="List of streams to evaluate.",
    )
    parser.add_argument(
        "-c",
        "--channels",
        type=str,
        nargs="*",
        default=None,
        help="List of channels to evaluate.",
    )
    parser.add_argument(
        "-e", "--epoch", type=int, default=0, help="Epoch number of inference run."
    )
    parser.add_argument(
        "-r", "--rank", type=int, default=0, help="Rank of inference run."
    )

    args = parser.parse_args()

    if args.save_dir is None:
        save_dir = _DEFAULT_RESULT_PATH / args.run_id
    else:
        save_dir = args.save_dir

    fast_evaluation(
        args.run_id,
        args.metrics,
        save_dir,
        args.results_dir,
        args.streams,
        args.channels,
        args.epoch,
        args.rank,
    )
