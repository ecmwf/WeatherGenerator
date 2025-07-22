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

import numpy as np
import xarray as xr
from score import VerifiedData, get_score
from utils import to_list

from weathergen.common.io import ZarrIO

_logger = logging.getLogger(__name__)

_REPO_ROOT = Path(
    __file__
).parent.parent.parent.parent.parent.parent  # TODO use importlib for resources
_DEFAULT_RESULT_PATH = _REPO_ROOT / "results"

### TODO:
# - Make use of config file from inference run
# - Add support a posteriori aggregation of metrics
# - Convert fsteps into lead time hours


### Auxiliary functions
def peek_tar_channels(zio: ZarrIO, stream: str) -> list[str]:
    """
    Peek the channels of a target stream in a ZarrIO object.

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
    _logger.debug(f"Peeked channels for stream {stream}: {channels}")

    return channels


def calc_scores_per_stream(
    zio: ZarrIO,
    stream: str,
    metrics: list[str],
    channels: str | list[str] | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
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
    channels : str | list[str] | None, optional
        A list of channel names to restrict the evaluation to. If None, all channels are used.
    Returns
    -------
    metric_stream : xr.DataArray
        An xarray DataArray containing the computed metrics for the specified stream.
    """
    # Get stream-specific information
    forecast_steps = zio.forecast_steps
    nmetrics, nforecast_steps = len(metrics), len(forecast_steps)
    # TODO: Avoid conversion to integer and sorting
    samples = sorted([int(sample) for sample in zio.samples])
    nsamples = len(samples)

    channels_stream = peek_tar_channels(zio, stream)
    # filter channels if provided
    channels = (
        [ch for ch in channels_stream if ch in to_list(channels)]
        if channels
        else channels_stream
    )
    nchannels = len(channels)

    # initialize the DataArray to store metrics
    # TODO: Rather concatenate list of DataArrays than initializing empty arrays.
    # TODO: Support lazy initialization (with dask) and subsequent processing
    #       for very large number of forecast steps.
    metric_stream = xr.DataArray(
        np.full(
            (nsamples, nforecast_steps, nchannels, nmetrics),
            np.nan,
        ),
        coords={
            "sample": samples,
            "forecast_step": forecast_steps,
            "channel": channels,
            "metric": metrics,
        },
    )

    points_per_sample = xr.DataArray(
        np.full((nforecast_steps, nsamples), np.nan),
        coords={"forecast_step": forecast_steps, "sample": samples},
        dims=("forecast_step", "sample"),
        name=f"points_per_sample_{stream}",
    )

    for fstep in forecast_steps:
        targets, preds = [], []
        pps = []
        _logger.info(f"Processing forecast_step {fstep} of stream {stream}...")
        for sample in samples:
            out = zio.get_data(sample, stream, fstep)
            target, pred = out.target.as_xarray(), out.prediction.as_xarray()

            targets.append(target.squeeze())
            preds.append(pred.squeeze())
            pps.append(len(target.ipoint))

        # Concatenate targets and predictions along the 'ipoint' dimension and verify the data
        _logger.debug(
            f"Concatenating targets and predictions for stream {stream}, forecast_step {fstep}..."
        )
        targets_all, preds_all = (
            xr.concat(targets, dim="ipoint"),
            xr.concat(preds, dim="ipoint"),
        )

        if channels != channels_stream:
            _logger.debug(
                f"Restricting targets and predictions to channels {channels} for stream {stream}, forecast_step {fstep}..."
            )
            targets_all = targets_all.sel(channel=channels)
            preds_all = preds_all.sel(channel=channels)

        _logger.debug(f"Verifying data for stream {stream}, forecast_step {fstep}...")
        score_data = VerifiedData(preds_all, targets_all)

        # Build up computation graphs for all metrics
        _logger.debug(
            f"Build computation graphs for metrics for stream {stream}, forecast_step {fstep}..."
        )
        combined_metrics = [
            get_score(score_data, metric, agg_dims="ipoint", group_by_coord="sample")
            for metric in metrics
        ]
        combined_metrics = xr.concat(combined_metrics, dim="metric")
        combined_metrics["metric"] = metrics

        # Do computation and store the computed metrics in the DataArray
        _logger.debug("Running computation of metrics...")
        metric_stream.loc[{"forecast_step": fstep}] = combined_metrics.compute()
        points_per_sample.loc[{"forecast_step": fstep}] = np.array(pps)
        _logger.info(f"Computed metrics for forecast_step {fstep} of stream {stream}.")

    return metric_stream, points_per_sample


def metric_list_to_dict(
    metrics_list: list[xr.DataArray],
    npoints_sample_list: list[xr.DataArray],
    streams: list[str],
) -> list[dict]:
    """
    Convert a list of xarray DataArrays containing metrics and a corresponding list of DataArrays containing
    the number of points per sample into a list of dictionaries.

    NOTE:
    This function is not used right now, but kept for future reference.

    Parameters
    ----------
    metrics_list : list[xr.DataArray]
        A list of xarray DataArrays, each containing metrics for a specific stream.
    npoints_sample_list : list[xr.DataArray]
        A list of xarray DataArrays, each containing the number of points per sample for the corresponding stream.
    streams : list[str]
        A list of stream names corresponding to the DataArrays in metrics_list and npoints_sample_list.

    Returns
    -------
    result : list[dict]
        A list of dictionaries where each dictionary contains the stream name, sample ID, forecast step,
        channel name, metrics, and the number of points for that sample.
    """

    result = []

    assert len(metrics_list) == len(npoints_sample_list) == len(streams), (
        f"The lengths of metrics_list ({len(metrics_list)}), npoints_sample_list ({len(npoints_sample_list)}), "
        + f"and streams ({len(streams)}) must be the same."
    )

    for stream, metrics_da, points_da in zip(
        streams, metrics_list, npoints_sample_list, strict=False
    ):
        metric_names = metrics_da.coords["metric"].values
        samples = metrics_da.coords["sample"].values
        steps = metrics_da.coords["forecast_step"].values
        channels = metrics_da.coords["channel"].values

        for sample in samples:
            for step in steps:
                n_points = int(
                    points_da.sel({"forecast_step": step, "sample": sample}).values
                )
                for channel in channels:
                    record = {
                        "stream": stream,
                        "sample": int(sample),
                        "forecast_step": str(step),
                        "channel": str(channel),
                        "metrics": {},
                        "n_points": n_points,
                    }

                    metrics_samples = metrics_da.sel(
                        {"sample": sample, "forecast_step": step, "channel": channel}
                    ).values
                    for m_idx in range(len(metric_names)):
                        record["metrics"][str(metric_names[m_idx])] = float(
                            metrics_samples[m_idx]
                        )

                    result.append(record)

    return result


def metric_list_to_json(
    metrics_list: list[xr.DataArray],
    npoints_sample_list: list[xr.DataArray],
    streams: list[str],
    metric_dir: Path,
    run_id: str,
    epoch: int,
):
    """
    Write the evaluation results collected in a list of xarray DataArrays for the metrics to to stream- and metric-specific JSON files.

    Parameters
    ----------
    metrics_list : list[xr.DataArray]
        A list of xarray DataArrays, each containing metrics for a specific stream.
    npoints_sample_list : list[xr.DataArray]
        A list of xarray DataArrays, each containing the number of points per sample for the corresponding stream.
    streams : list[str]
        A list of stream names corresponding to the DataArrays in metrics_list and npoints_sample_list.
    metric_dir : Path
        The directory where the JSON files will be saved.
    run_id : str
        The ID of the inference run to evaluate.
    epoch : int
        The epoch number of the inference run.
    """

    assert len(metrics_list) == len(npoints_sample_list) == len(streams), (
        "The lengths of metrics_list, npoints_sample_list, and streams must be the same."
    )

    # Ensure the save directory exists
    metric_dir.mkdir(parents=True, exist_ok=True)

    for s_idx, stream in enumerate(streams):
        metrics_stream, npoints_sample_stream = (
            metrics_list[s_idx],
            npoints_sample_list[s_idx],
        )

        _logger.debug(f"Processing metrics from stream {stream}...")

        for metric in metrics_stream.coords["metric"].values:
            _logger.debug(f"Processing metric {metric} of stream {stream}...")

            # Select the metric data for the current stream and convert to a xarray Dataset
            metric_now = metrics_stream.sel(metric=metric)
            metric_ds = xr.Dataset(
                {"metric": metric_now, "n_datapoints": npoints_sample_stream}
            )

            # Convert the Dataset to a dictionary
            metric_dict = metric_ds.to_dict()

            # Save the results to a JSON file
            save_path = metric_dir / f"{run_id}_{stream}_{metric}_epoch{epoch:05d}.json"

            _logger.info(f"Saving results to {save_path}")
            with open(save_path, "w") as f:
                json.dump(metric_dict, f, indent=4)

    _logger.info(
        f"Saved all results of inference run {run_id} - epoch {epoch:d} successfully to {metric_dir}."
    )


def fast_evaluation(
    run_id: str,
    metrics: list[str],
    metric_dir: Path,
    results_dir: Path = _DEFAULT_RESULT_PATH,
    streams: str | list[str] | None = None,
    channels: str | list[str] | None = None,
    epoch: int = 0,
    rank: int = 0,
):
    """
    Perform fast evaluation of a run using the specified metrics and save the results.

    Parameters
    ----------
    run_id : str
        The ID of the inference run to evaluate.
    metrics : list[str]
        A list of metric names to evaluate.
    metric_dir : Path
        The directory where the JSON-file with the metric results will be saved.
    """

    # get path to zarr storage
    results_zarr = (
        results_dir / run_id / f"validation_epoch{epoch:05d}_rank{rank:04d}.zarr"
    )

    if not results_zarr.exists():
        raise FileNotFoundError(f"Results zarr file not found: {results_zarr}")

    all_metric_streams = []
    points_per_sample_streams = []

    # Open the ZarrIO object to access the data
    _logger.info(f"Loading inference data from{results_zarr}")

    with ZarrIO(results_zarr) as zio:
        streams = streams or zio.streams

        for stream in streams:
            _logger.info(f"Processing stream {stream}...")

            metric_stream, pps = calc_scores_per_stream(zio, stream, metrics, channels)
            all_metric_streams.append(metric_stream)
            points_per_sample_streams.append(pps)

    _logger.info(
        f"Finished computing metric scores for all streams. Total streams processed: {len(all_metric_streams)}"
    )

    # Save the metrics to individual JSON files
    _logger.info("Saving metrics to individual JSON files...")
    metric_list_to_json(
        all_metric_streams,
        points_per_sample_streams,
        streams,
        metric_dir,
        run_id,
        epoch,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fast evaluation of WeatherGenerator inference runs."
    )

    parser.add_argument(
        "-id", "--run-id", type=str, help="The ID of the inference run to evaluate."
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
        "-md",
        "--metric-dir",
        type=Path,
        default=None,
        help="Directory to save the metric results.",
    )
    parser.add_argument(
        "-rd",
        "--results-dir",
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
        nargs="+",
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

    logging.basicConfig(level=logging.DEBUG)

    if args.metric_dir is None:
        metric_dir = _DEFAULT_RESULT_PATH / args.run_id
    else:
        metric_dir = args.metric_dir

    fast_evaluation(
        args.run_id,
        args.metrics,
        metric_dir,
        args.results_dir,
        args.streams,
        args.channels,
        args.epoch,
        args.rank,
    )
