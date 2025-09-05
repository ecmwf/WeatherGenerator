# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import omegaconf as oc
import xarray as xr
from tqdm import tqdm

from weathergen.common.io import ZarrIO
from weathergen.evaluate.plotter import LinePlots, Plotter
from weathergen.evaluate.score import VerifiedData, get_score
from weathergen.evaluate.score_utils import RegionBoundingBox, to_list
from weathergen.evaluate.plot_utils import collect_streams, collect_channels, plot_metric_region
from weathergen.utils.config import Config

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


@dataclass
class WeatherGeneratorOutput:
    target: dict
    prediction: dict
    points_per_sample: xr.DataArray | None


def get_data(
    cfg: dict,
    results_dir: Path,
    stream: str,
    region: str = "global",
    samples: list[int] = None,
    fsteps: list[str] = None,
    channels: list[str] = None,
    return_counts: bool = False,
) -> WeatherGeneratorOutput:
    """
    Retrieve prediction and target data for a given run from the Zarr store.

    Parameters
    ----------
    cfg :
        Configuration dictionary containing all information for the evaluation.
    results_dir : Path
        Directory where the inference results are stored. Expected scheme `<results_base_dir>/<run_id>`.
    stream :
        Stream name to retrieve data for.
    region :
        Region name to retrieve data for.
    samples :
        List of sample indices to retrieve. If None, all samples are retrieved.
    fsteps :
        List of forecast steps to retrieve. If None, all forecast steps are retrieved.
    channels :
        List of channel names to retrieve. If None, all channels are retrieved.
    return_counts :
        If True, also return the number of points per sample.

    Returns
    -------
    WeatherGeneratorOutput
        A dataclass containing:
        - target: Dictionary of xarray DataArrays for targets, indexed by forecast step.
        - prediction: Dictionary of xarray DataArrays for predictions, indexed by forecast step.
        - points_per_sample: xarray DataArray containing the number of points per sample, if `return_counts` is True.
    """
    run_id = results_dir.name
    run = cfg.run_ids[run_id]

    fname_zarr = results_dir.joinpath(
        f"validation_epoch{run['epoch']:05d}_rank{run['rank']:04d}.zarr"
    )

    if not fname_zarr.exists() or not fname_zarr.is_dir():
        _logger.error(f"Zarr file {fname_zarr} does not exist or is not a directory.")
        raise FileNotFoundError(
            f"Zarr file {fname_zarr} does not exist or is not a directory."
        )

    bbox = RegionBoundingBox.from_region_name(region)

    with ZarrIO(fname_zarr) as zio:
        zio_forecast_steps = zio.forecast_steps
        stream_dict = run.streams[stream]
        all_channels = peek_tar_channels(zio, stream, zio_forecast_steps[0])
        _logger.info(f"RUN {run_id}: Processing stream {stream}...")

        fsteps = zio_forecast_steps if fsteps is None else fsteps
        # TODO: Avoid conversion of fsteps and sample to integers (as obtained from the ZarrIO)
        fsteps = sorted([int(fstep) for fstep in fsteps])
        samples = sorted(
            [int(sample) for sample in zio.samples] if samples is None else samples
        )
        channels = channels or stream_dict.get("channels", all_channels)
        channels = to_list(channels)

        da_tars, da_preds = [], []

        if return_counts:
            points_per_sample = xr.DataArray(
                np.full((len(fsteps), len(samples)), np.nan),
                coords={"forecast_step": fsteps, "sample": samples},
                dims=("forecast_step", "sample"),
                name=f"points_per_sample_{stream}",
            )
        else:
            points_per_sample = None

        fsteps_final = []

        for fstep in fsteps:
            _logger.info(f"RUN {run_id} - {stream}: Processing fstep {fstep}...")
            da_tars_fs, da_preds_fs = [], []
            pps = []

            for sample in tqdm(
                samples, desc=f"Processing {run_id} - {stream} - {fstep}"
            ):
                out = zio.get_data(sample, stream, fstep)
                target, pred = out.target.as_xarray(), out.prediction.as_xarray()

                if region != "global":
                    _logger.debug(
                        f"Applying bounding box mask for region '{region}' to targets and predictions..."
                    )
                    target = bbox.apply_mask(target)
                    pred = bbox.apply_mask(pred)

                npoints = len(target.ipoint)
                if npoints == 0:
                    _logger.info(
                        f"Skipping {stream} sample {sample} forecast step: {fstep}. Dataset is empty."
                    )
                    continue

                da_tars_fs.append(target.squeeze())
                da_preds_fs.append(pred.squeeze())
                pps.append(npoints)

            if len(da_tars_fs) > 0:
                fsteps_final.append(fstep)

            _logger.debug(
                f"Concatenating targets and predictions for stream {stream}, forecast_step {fstep}..."
            )

            if da_tars_fs:
                da_tars_fs = xr.concat(da_tars_fs, dim="ipoint")
                da_preds_fs = xr.concat(da_preds_fs, dim="ipoint")

                if set(channels) != set(all_channels):
                    _logger.debug(
                        f"Restricting targets and predictions to channels {channels} for stream {stream}..."
                    )
                    available_channels = da_tars_fs.channel.values
                    existing_channels = [
                        ch for ch in channels if ch in available_channels
                    ]
                    if len(existing_channels) < len(channels):
                        _logger.warning(
                            f"The following channels were not found: {list(set(channels) - set(existing_channels))}. Skipping them."
                        )

                    da_tars_fs = da_tars_fs.sel(channel=existing_channels)
                    da_preds_fs = da_preds_fs.sel(channel=existing_channels)

                da_tars.append(da_tars_fs)
                da_preds.append(da_preds_fs)
            if return_counts:
                points_per_sample.loc[{"forecast_step": fstep}] = np.array(pps)
        
        # Safer than a list
        da_tars = {fstep: da for fstep, da in zip(fsteps_final, da_tars, strict=False)}
        da_preds = {
            fstep: da for fstep, da in zip(fsteps_final, da_preds, strict=False)
        }

        return WeatherGeneratorOutput(
            target=da_tars, prediction=da_preds, points_per_sample=points_per_sample
        )


def calc_scores_per_stream(
    cfg: dict, results_dir: Path, stream: str, region: str, metrics: list[str]
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Calculate scores for a given run and stream using the specified metrics.

    Parameters
    ----------
    cfg :
        Configuration dictionary containing all information for the evaluation.
    results_dir : Path
        Directory where the results are stored.
        Expected scheme `<results_base_dir>/<run_id>`.
    stream :
        Stream name to calculate scores for.
    region :
        Region name to calculate scores for.
    metrics :
        List of metric names to calculate.

    Returns
    -------
    Tuple of xarray DataArray containing the scores and the number of points per sample.
    """
    run_id = results_dir.name

    _logger.info(
        f"RUN {run_id} - {stream}: Calculating scores for metrics {metrics}..."
    )

    samples = cfg.evaluation.get("sample", None)
    fsteps = cfg.evaluation.get("forecast_step", None)
    channels = cfg.get("channels")

    if samples == "all":
        samples = None

    if fsteps == "all":
        fsteps = None

    output_data = get_data(cfg, results_dir, stream, region=region, return_counts=True)

    da_preds = output_data.prediction
    da_tars = output_data.target
    points_per_sample = output_data.points_per_sample

    # get coordinate information from retrieved data

    fsteps = [int(k) for k in da_tars.keys()]
    first_da = list(da_preds.values())[0]

    # TODO: improve the way we handle samples.
    samples = list(np.atleast_1d(np.unique(first_da.sample.values)))
    channels = list(np.atleast_1d(first_da.channel.values))

    metric_list = []

    metric_stream = xr.DataArray(
        np.full(
            (len(samples), len(fsteps), len(channels), len(metrics)),
            np.nan,
        ),
        coords={
            "sample": samples,
            "forecast_step": fsteps,
            "channel": channels,
            "metric": metrics,
        },
    )

    for (fstep, tars), (_, preds) in zip(
        da_tars.items(), da_preds.items(), strict=False
    ):
        _logger.debug(f"Verifying data for stream {stream}...")

        if preds.ipoint.size > 0:
            score_data = VerifiedData(preds, tars)

            # Build up computation graphs for all metrics
            _logger.debug(
                f"Build computation graphs for metrics for stream {stream}..."
            )

            combined_metrics = [
                get_score(
                    score_data, metric, agg_dims="ipoint", group_by_coord="sample"
                )
                for metric in metrics
            ]

            combined_metrics = xr.concat(combined_metrics, dim="metric")
            combined_metrics["metric"] = metrics

            _logger.debug(f"Running computation of metrics for stream {stream}...")
            combined_metrics = combined_metrics.compute()
            combined_metrics = scalar_coord_to_dim(combined_metrics, "channel")
        else:
            # depending on the datset, there might be no data (e.g. no CERRA in southern hemisphere region)
            _logger.warning(
                f"No data available for stream {stream} at forecast step {fstep} in region {region}. Skipping metrics calculation."
            )
            continue

        metric_list.append(combined_metrics)

        metric_stream.loc[{"forecast_step": int(fstep)}] = combined_metrics

    _logger.info(f"Scores for run {run_id} - {stream} calculated successfully.")

    metric_stream = xr.concat(metric_list, dim="forecast_step")
    metric_stream = metric_stream.assign_coords({"forecast_step": fsteps})

    return metric_stream, points_per_sample


def plot_data(
    cfg: dict,
    run_config: oc.DictConfig,
    results_dir: Path,
    plot_dir: Path,
    stream: str,
) -> list[str]:
    """
    Plot the data for a given run and stream.

    Parameters
    ----------
    cfg :
        Configuration dictionary containing all information for the evaluation.
        Must provide a stream configuration under `cfg['run_ids'][run_id]["streams"][stream]`.
    run_config :
        Configuration for the run, including stream information.
    results_dir :
        Directory where the inference results are stored.
        Expected scheme `<results_base_dir>/<run_id>`.
    plot_base_dir :
        Base directory where the plots will be saved.
    stream :
        Stream name to plot data for.
    Returns
    -------
    List of plot names generated during the plotting process.
    """
    run_id = results_dir.name

    # get stream dict from evaluation config (assumed to be part of cfg at this point)
    stream_dict = cfg["run_ids"][run_id]["streams"][stream]

    # handle plotting settings
    plot_settings = stream_dict.get("plotting", {})

    # return early if no plotting is requested
    if not (
        plot_settings
        and (
            plot_settings.get("plot_maps", False)
            or plot_settings.get("plot_histograms", False)
            or plot_settings.get("plot_animations", False)
        )
    ):
        return

    # get plotter configuration
    plotter_cfg = {
        "image_format": cfg.get("image_format", "png"),
        "dpi_val": cfg.get("dpi_val", 300),
        "fig_size": cfg.get("fig_size", (8, 10)),
        "plot_subtimesteps": get_stream_attr(
            run_config, stream, "tokenize_spacetime", False
        ),
    }

    plotter = Plotter(plotter_cfg, plot_dir)

    plot_samples = plot_settings.get("sample", None)
    plot_fsteps = plot_settings.get("forecast_step", None)
    plot_chs = stream_dict.get("channels")

    # Check if maps should be plotted and handle configuration if provided
    plot_maps = plot_settings.get("plot_maps", False)
    if not isinstance(plot_maps, bool):
        raise TypeError("plot_maps must be a boolean.")

    if isinstance(cfg.get("global_plotting_options", False), oc.dictconfig.DictConfig):
        if isinstance(
            cfg.global_plotting_options.get(stream), oc.dictconfig.DictConfig
        ):
            maps_config = cfg.global_plotting_options.get(stream)
        else:
            cfg["global_plotting_options"][stream] = {}
            maps_config = cfg.global_plotting_options.get(stream)
    else:
        cfg["global_plotting_options"] = {stream: {}}
        maps_config = cfg.global_plotting_options.get(stream)

    # Check if histograms should be plotted
    plot_histograms = plot_settings.get("plot_histograms", False)
    if not isinstance(plot_histograms, bool):
        raise TypeError("plot_histograms must be a boolean.")

    plot_animations = plot_settings.get("plot_animations", False)
    if not isinstance(plot_animations, bool):
        raise TypeError("plot_animations must be a boolean.")

    if plot_fsteps == "all":
        plot_fsteps = None

    if plot_samples == "all":
        plot_samples = None

    model_output = get_data(
        cfg,
        results_dir,
        stream,
        samples=plot_samples,
        fsteps=plot_fsteps,
        channels=plot_chs,
    )

    da_tars = model_output.target
    da_preds = model_output.prediction

    if not da_tars:
        _logger.info(f"Skipping Plot Data for {stream}. Targets are empty.")
        return

    maps_config = common_ranges(da_tars, da_preds, plot_chs, maps_config)

    plot_names = []

    plot_names = []

    for (fstep, tars), (_, preds) in zip(
        da_tars.items(), da_preds.items(), strict=False
    ):
        plot_chs = list(np.atleast_1d(tars.channel.values))
        plot_samples = list(np.unique(tars.sample.values))

        for sample in tqdm(
            plot_samples, desc=f"Plotting {run_id} - {stream} - fstep {fstep}"
        ):
            plots = []

            data_selection = {
                "sample": sample,
                "stream": stream,
                "forecast_step": fstep,
            }

            if plot_maps:
                map_tar = plotter.create_maps_per_sample(
                    tars, plot_chs, data_selection, "targets", maps_config
                )

                map_pred = plotter.create_maps_per_sample(
                    preds, plot_chs, data_selection, "preds", maps_config
                )
                plots.extend([map_tar, map_pred])

            if plot_histograms:
                h = plotter.create_histograms_per_sample(
                    tars, preds, plot_chs, data_selection
                )
                plots.append(h)

            plotter = plotter.clean_data_selection()

            plot_names.append(plots)

    if plot_animations:
        plot_fsteps = da_tars.keys()
        h = plotter.animation(
            plot_samples, plot_fsteps, plot_chs, data_selection, "preds"
        )
        h = plotter.animation(
            plot_samples, plot_fsteps, plot_chs, data_selection, "targets"
        )

    return plot_names


def metric_list_to_json(
    metrics_list: list[xr.DataArray],
    npoints_sample_list: list[xr.DataArray],
    streams: list[str],
    region: str,
    metric_dir: Path,
    run_id: str,
    epoch: int,
):
    """
    Write the evaluation results collected in a list of xarray DataArrays for the metrics
    to stream- and metric-specific JSON files.

    Parameters
    ----------
    metrics_list :
        Metrics per stream.
    npoints_sample_list :
        Number of points per sample per stream.
    streams :
        Stream names.
    region :
        Region name.
    metric_dir :
        Output directory.
    run_id :
        Identifier of the inference run.
    epoch :
        Epoch number.
    """
    assert len(metrics_list) == len(npoints_sample_list) == len(streams), (
        "The lengths of metrics_list, npoints_sample_list, and streams must be the same."
    )

    metric_dir.mkdir(parents=True, exist_ok=True)

    for s_idx, stream in enumerate(streams):
        metrics_stream, npoints_sample_stream = (
            metrics_list[s_idx],
            npoints_sample_list[s_idx],
        )

        for metric in metrics_stream.coords["metric"].values:
            metric_now = metrics_stream.sel(metric=metric)

            # Save as individual DataArray, not Dataset
            metric_now.attrs["npoints_per_sample"] = (
                npoints_sample_stream.values.tolist()
            )
            metric_dict = metric_now.to_dict()

            # Match the expected filename pattern
            save_path = (
                metric_dir
                / f"{run_id}_{stream}_{region}_{metric}_epoch{epoch:05d}.json"
            )

            _logger.info(f"Saving results to {save_path}")
            with open(save_path, "w") as f:
                json.dump(metric_dict, f, indent=4)

    _logger.info(
        f"Saved all results of inference run {run_id} - epoch {epoch:d} successfully to {metric_dir}."
    )


def retrieve_metric_from_json(
    metric_dir: str, run_id: str, stream: str, region: str, metric: str, epoch: int
):
    """
    Retrieve the score for a given run, stream, metric, epoch, and rank from a JSON file.

    Parameters
    ----------
    metric_dir :
        Directory where JSON files are stored.
    run_id :
        Run identifier.
    stream :
        Stream name.
    region :
        Region name.
    metric :
        Metric name.
    epoch :
        Epoch number.

    Returns
    -------
    xr.DataArray
        The metric DataArray.
    """
    score_path = (
        Path(metric_dir) / f"{run_id}_{stream}_{region}_{metric}_epoch{epoch:05d}.json"
    )
    _logger.debug(f"Looking for: {score_path}")
    if score_path.exists():
        with open(score_path) as f:
            data_dict = json.load(f)
            return xr.DataArray.from_dict(data_dict)
    else:
        raise FileNotFoundError(f"File {score_path} not found in the archive.")


def plot_summary(cfg: dict, scores_dict: dict, summary_dir: Path):
    """
    Plot summary of the evaluation results.
    This function is a placeholder for future implementation.

    Parameters
    ----------
    cfg :
        Configuration dictionary containing all information for the evaluation.
    scores_dict :
        Dictionary containing scores for each metric and stream.
    """
    _logger.info("Plotting summary of evaluation results...")

    runs = cfg.run_ids
    metrics = cfg.evaluation.metrics
    print_summary = cfg.evaluation.get("print_summary", False)
    regions = cfg.evaluation.get("regions", ["global"])

    plotter = LinePlots(cfg, summary_dir)

    for region in regions:
        for metric in metrics:
            plot_metric_region(metric, region, runs, scores_dict, plotter, print_summary)


############# Utility functions ############


def common_ranges(
    data_tars: list[dict],
    data_preds: list[dict],
    plot_chs: list[str],
    maps_config: oc.dictconfig.DictConfig,
) -> oc.dictconfig.DictConfig:
    """
    Calculate common ranges per stream and variables.

    Parameters
    ----------
    data_tars :
        the (target) list of dictionaries with the forecasteps and respective xarray
    data_preds :
        the (prediction) list of dictionaries with the forecasteps and respective xarray
    plot_chs:
        the variables to be plotted as given by the configuration file
    maps_config:
        the global plotting configuration
    Returns
    -------
    maps_config :
        the global plotting configuration with the ranges added and included for each variable (and for each stream).
    """

    for var in plot_chs:
        if var in maps_config:
            if not isinstance(maps_config[var].get("vmax"), (int | float)):
                list_max = calc_bounds(data_tars, data_preds, var, "max")

                maps_config[var].update({"vmax": float(max(list_max))})

            if not isinstance(maps_config[var].get("vmin"), (int | float)):
                list_min = calc_bounds(data_tars, data_preds, var, "min")

                maps_config[var].update({"vmin": float(min(list_min))})

        else:
            list_max = calc_bounds(data_tars, data_preds, var, "max")

            list_min = calc_bounds(data_tars, data_preds, var, "min")

            maps_config.update(
                {var: {"vmax": float(max(list_max)), "vmin": float(min(list_min))}}
            )

    return maps_config


def calc_val(x: xr.DataArray, bound: str) -> list[float]:
    """
    Calculate the maximum or minimum value per variable for all forecasteps.
    Parameters
    ----------
    x :
        the xarray DataArray with the forecasteps and respective values
    bound :
        the bound to be calculated, either "max" or "min"
    Returns
    -------
        a list with the maximum or minimum values for a specific variable.
    """
    if bound == "max":
        return x.max(dim=("ipoint")).values
    elif bound == "min":
        return x.min(dim=("ipoint")).values
    else:
        raise ValueError("bound must be either 'max' or 'min'")


def calc_bounds(
    data_tars,
    data_preds,
    var,
    bound,
):
    """
    Calculate the minimum and maximum values per variable for all forecasteps for both targets and predictions

    Parameters
    ----------
    data_tars :
        the (target) list of dictionaries with the forecasteps and respective xarray
    data_preds :
        the (prediction) list of dictionaries with the forecasteps and respective xarray
    Returns
    -------
    list_bound :
        a list with the maximum or minimum values for a specific variable.
    """
    list_bound = []
    for da_tars, da_preds in zip(data_tars.values(), data_preds.values(), strict=False):
        list_bound.extend(
            (
                calc_val(da_tars.where(da_tars.channel == var, drop=True), bound),
                calc_val(da_preds.where(da_preds.channel == var, drop=True), bound),
            )
        )

    return list_bound


def peek_tar_channels(zio: ZarrIO, stream: str, fstep: int = 0) -> list[str]:
    """
    Peek the channels of a target stream in a ZarrIO object.

    Parameters
    ----------
    zio :
        The ZarrIO object containing the tar stream.
    stream :
        The name of the tar stream to peek.
    fstep :
        The forecast step to peek. Default is 0.
    Returns
    -------
    channels :
        A list of channel names in the tar stream.
    """
    if not isinstance(zio, ZarrIO):
        raise TypeError("zio must be an instance of ZarrIO")

    dummy_out = zio.get_data(0, stream, fstep)
    channels = dummy_out.target.channels
    _logger.debug(f"Peeked channels for stream {stream}: {channels}")

    return channels


def scalar_coord_to_dim(da: xr.DataArray, name: str, axis: int = -1) -> xr.DataArray:
    """
    Convert a scalar coordinate to a dimension in an xarray DataArray.
    If the coordinate is already a dimension, it is returned unchanged.

    Parameters
    ----------
    da : xarray.DataArray
        The DataArray to modify.
    name : str
        The name of the coordinate to convert.
    axis : int, optional
        The axis along which to expand the dimension. Default is -1 (last axis).
    Returns
    -------
    xarray.DataArray
        The modified DataArray with the scalar coordinate converted to a dimension.
    """
    if name in da.dims:
        return da  # already a dimension
    if name in da.coords and da.coords[name].ndim == 0:
        val = da.coords[name].item()
        da = da.drop_vars(name)
        da = da.expand_dims({name: [val]}, axis=axis)
    return da


def get_stream_attr(config: Config, stream_name: str, key: str, default=None):
    """
    Get the value of a key for a specific stream from the a model config.

    Parameters:
    ------------
        config: dict
            The full configuration dictionary.
        stream_name: str
            The name of the stream (e.g. 'ERA5').
        key: str
            The key to look up (e.g. 'tokenize_spacetime').
        default: Optional
            Value to return if not found (default: None).

    Returns:
        The parameter value if found, otherwise the default.
    """
    for stream in config.get("streams", []):
        if stream.get("name") == stream_name:
            return stream.get(key, default)
    return default
